# Copyright 2024 The TensorFlow Recommenders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 The TensorFlow Recommenders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements `Cross` Layer, the cross layer in Deep & Cross Network (DCN)."""

from typing import Union, Text, Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class MultiLayerDCN(tf.keras.layers.Layer):
  """Cross Layer in Deep & Cross Network to learn explicit feature interactions."""

  def __init__(
      self,
      projection_dim: Optional[int] = 1,
      num_layers: Optional[int] = 3,
      use_bias: bool = True,
      kernel_initializer: Union[Text, tf.keras.initializers.Initializer] = "he_uniform",
      bias_initializer: Union[Text, tf.keras.initializers.Initializer] = "zeros",
      kernel_regularizer: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
      **kwargs
  ):
    super(MultiLayerDCN, self).__init__(**kwargs)
    self._projection_dim = projection_dim
    self._num_layers = num_layers
    self._use_bias = use_bias
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._input_dim = None
    self._supports_masking = True

  def build(self, input_shape):
    last_dim = input_shape[-1]
    self._u_kernels = []
    self._v_kernels = []
    self._biases = []
    
    for i in range(self._num_layers):
      self._u_kernels.append(
        self.add_weight(
            shape=(last_dim, self._projection_dim),
            initializer=tf.keras.initializers.VarianceScaling(
                mode='fan_avg', distribution='untruncated_normal'),
            regularizer=self._kernel_regularizer,
            trainable=True,
            name=f'u_kernel_{i}'
        )
      )
      self._v_kernels.append(
        self.add_weight(
            shape=(self._projection_dim, last_dim),
            initializer=tf.keras.initializers.VarianceScaling(
                mode='fan_avg', distribution='untruncated_normal'),
            regularizer=self._kernel_regularizer,
            trainable=True,
            name=f'v_kernel_{i}'
        )
      )
      if self._use_bias:
        self._biases.append(
            self.add_weight(
                shape=(last_dim,),
                initializer='zeros',
                trainable=True,
                name=f'bias_{i}'
            )
        )
        

    self.built = True

  def call(self, x0: tf.Tensor) -> tf.Tensor:
    xl = x0
    for i in range(self._num_layers):
      u_output = tf.matmul(xl, self._u_kernels[i])
      v_output = tf.matmul(u_output, self._v_kernels[i])
      if self._use_bias:
          v_output += self._biases[i]
      xl = x0 * v_output + xl
    return xl

  def get_config(self):
    config = {
      "projection_dim": self._projection_dim,
      "num_layers": self._num_layers,
      "use_bias": self._use_bias,
      "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
      "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
      "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
      "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _clone_initializer(initializer):
    return initializer.__class__.from_config(initializer.get_config())
  