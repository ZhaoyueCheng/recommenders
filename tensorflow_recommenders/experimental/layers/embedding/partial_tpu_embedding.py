"""Embedding layer for the Ranking model."""

from typing import Dict, Optional, Union, List

import tensorflow as tf

from tensorflow_recommenders.layers.embedding.tpu_embedding_layer import TPUEmbedding

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class PartialTPUEmbedding(tf.keras.layers.Layer):
  """Partial TPU Embedding layer.

  This layer is composed  of `tfrs.layers.embedding.TPUEmbedding` and
  `tf.keras.layers.Embedding` embedding layers. When training on TPUs, it is
  preferable to use TPU Embedding layers for large tables (as they are sharded
  accross TPU cores) and Keras embedding layer for small tables.
  For tables with vocab sizes less than `size_threshold` a Keras embedding
  layer will be used, above that threshold a TPU embedding layer will be used.

  This layer will be applied on a dictionary of feature_name, categorical_tensor
  pairs and return a dictionary of string-to-tensor of feature_name,
  embedded_value pairs.
  """

  def __init__(self,
               feature_config,
               optimizer: tf.keras.optimizers.Optimizer,
               pipeline_execution_with_tensor_core: bool = False,
               batch_size: Optional[int] = None,
               size_threshold: Optional[int] = 10_000,
               multi_hot_sizes: Optional[List[int]] = None) -> None:
    """Initializes the embedding layer.

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An optimizer used for TPU embeddings.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old with potential correctness drawbacks). Set to True
        for improved performance.
      batch_size: If set, this will be used as the global batch size and
        override the autodetection of the batch size from the layer's input.
        This is necesarry if all inputs to the layer's call are SparseTensors.
      size_threshold: A threshold for table sizes below which a Keras embedding
        layer is used, and above which a TPU embedding layer is used.
        Set `size_threshold=0` to use TPU embedding for all tables and
        `size_threshold=None` to use only Keras embeddings.
    """
    super().__init__()

    tpu_feature_config = {}
    table_to_keras_emb = {}
    self._keras_embedding_layers = {}
    self._keras_table_combiners = {}
    self._keras_table_configs = []
    self._keras_embedding_layers_indices = {}
    self._keras_table_multi_hot_sizes = []

    for idx, (name, embedding_feature_config) in enumerate(feature_config.items()):
      table_config = embedding_feature_config.table
      if (
          size_threshold is not None
          and table_config.vocabulary_size > size_threshold
      ):
        # TPUEmbedding layer.
        tpu_feature_config[name] = embedding_feature_config
        continue

      # Keras layer.
      # Multiple features can reuse the same table.
      if table_config not in table_to_keras_emb:
        # table_to_keras_emb[table_config] = tf.keras.layers.Embedding(
        #     input_dim=table_config.vocabulary_size,
        #     output_dim=table_config.dim,
        #     embeddings_initializer=table_config.initializer or "uniform",
        # )
        self._keras_table_configs.append(table_config)

      self._keras_embedding_layers_indices[name] = (
          self._keras_table_configs.index(table_config)
      )
      self._keras_table_multi_hot_sizes.append(
          multi_hot_sizes[idx]
      )
      self._keras_embedding_layers[name] = self._keras_table_configs.index(table_config)

      # self._keras_table_combiners[name] = table_config.combiner

    self._total_vocab_size = sum(
        table_config.vocabulary_size
        for table_config in self._keras_table_configs
    )
    tf._logging.info(f"total_vocab_size: {self._total_vocab_size}")
    self._keras_emb_offsets = [0]
    for table_config in self._keras_table_configs:
      self._keras_emb_offsets.append(
          self._keras_emb_offsets[-1] + table_config.vocabulary_size
      )
    for key in self._keras_embedding_layers_indices:
      self._keras_embedding_layers_indices[key] = self._keras_emb_offsets[
          self._keras_embedding_layers_indices[key]
      ]


    tf._logging.info(f"offsets: {self._keras_emb_offsets}")
    self.keras_embedding_layer = tf.keras.layers.Embedding(
        input_dim=self._total_vocab_size,
        output_dim=self._keras_table_configs[0].dim,
        embeddings_initializer=self._create_custom_initializer(),
    )

    self._tpu_embedding = None
    if tpu_feature_config:
      self._tpu_embedding = TPUEmbedding(
          tpu_feature_config, optimizer, pipeline_execution_with_tensor_core
      )

  def _create_custom_initializer(self):
    """Creates a custom initializer for different segments of the embedding table."""
    def initializer(shape, dtype=None):
      embedding_matrix = tf.zeros(shape, dtype=dtype)

      for idx, table_config in enumerate(self._keras_table_configs):
        start_idx = self._keras_emb_offsets[idx]
        end_idx = self._keras_emb_offsets[idx + 1]
        
        # Use the table's specific initializer or default to uniform
        table_initializer = table_config.initializer or tf.keras.initializers.RandomUniform(
            minval=-tf.math.sqrt(1.0/table_config.vocabulary_size), maxval=tf.math.sqrt(1.0/table_config.vocabulary_size)
        )
        
        # Initialize the sub-matrix corresponding to this table
        sub_matrix = table_initializer([end_idx - start_idx, table_config.dim], dtype=dtype)
        embedding_matrix = tf.tensor_scatter_nd_update(
            embedding_matrix,
            indices=tf.range(start_idx, end_idx)[:, tf.newaxis],
            updates=sub_matrix
        )

      return embedding_matrix

    return initializer
      
  def call(self, inputs: Dict[str, Tensor]) -> Dict[str, tf.Tensor]:
    """Computes the output of the embedding layer.

    It expects a string-to-tensor (or SparseTensor/RaggedTensor) dict as input,
    and outputs a dictionary of string-to-tensor of feature_name, embedded_value
    pairs. Note that SparseTensor/RaggedTensor are only supported for
    TPUEmbedding and are not supported for Keras embeddings.

    Args:
      inputs: A string-to-tensor (or SparseTensor/RaggedTensor) dictionary.

    Returns:
      output: A dictionary of string-to-tensor of feature_name, embedded_value
        pairs.

    Raises:
      ValueError if no tf.Tensor is passed to a Keras embedding layer.
    """
    keras_emb_inputs = {
        key: val for key, val in inputs.items()
        if key in self._keras_embedding_layers_indices
    }
    tpu_emb_inputs = {
        key: val for key, val in inputs.items()
        if key not in self._keras_embedding_layers
    }

    indices = []
    output = {}
    # table stacking
    for key, val in keras_emb_inputs.items():
      if not isinstance(val, tf.Tensor):
        raise ValueError("Only tf.Tensor input is supported for Keras embedding"
                         f" layers, but got: {type(val)}")
      
      base_id = self._keras_embedding_layers_indices[key]
      x = base_id + val
      #tf._logging.info(f"baseid + val: {x}")
      indices.append(x)

    tf._logging.info(f"indices: {len(indices)}")
    tf._logging.info(f"indices[0]: {len(indices[0])}")
    full_keras_emb_outputs = self.keras_embedding_layer(tf.concat(indices, axis=1))
    tf._logging.info(f"full_keras_emb_outputs: {full_keras_emb_outputs.shape}")
    
    individual_outputs = tf.split(full_keras_emb_outputs, self._keras_table_multi_hot_sizes, axis=1) 
    tf._logging.info(f"individual_outputs: {len(individual_outputs)}")
    for idx, key in enumerate(keras_emb_inputs.keys()):
      output[key] = individual_outputs[idx]
      #tf._logging.info(f"output[key]: {output[key].shape}")
      output[key] = tf.reduce_sum(individual_outputs[idx], axis=1)
      #tf._logging.info(f"output[key]: {output[key].shape}")

    tf._logging.info(f"output: {output}")

    if self._tpu_embedding:
      tpu_emb_output_dict = self._tpu_embedding(tpu_emb_inputs)  # pylint: disable=[not-callable]
      output.update(tpu_emb_output_dict)
    return output

  @property
  def tpu_embedding(self) -> Optional[TPUEmbedding]:
    """Returns TPUEmbedding or `None` if only Keras embeddings are used."""
    return self._tpu_embedding

  # @property
  # def keras_embedding_layers(self) -> Dict[str, tf.keras.layers.Embedding]:
  #   """Returns a dictionary mapping feature names to Keras embedding layers."""
  #   return self._keras_embedding_layers