"""Keras-based positional embedding layer."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="mytf2")
class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  Example:
  ```python
  position_embedding = PositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```


  Arguments:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               **kwargs):
    super(PositionEmbedding, self).__init__(**kwargs)
    self._max_length = max_length
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)

  def get_config(self):
    config = {
        "max_length": self._max_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()

    if len(dimension_list) != 3:
      raise ValueError("PositionEmbedding expects a 3-dimensional input tensor "
               "of shape [batch, sequence, width], got "
               "{}".format(input_shape))
    seq_length = dimension_list[1]
    width = dimension_list[2]

    if self._max_length is not None:
      weight_sequence_length = self._max_length
    else:
      weight_sequence_length = seq_length
    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer,
    )
    super(PositionEmbedding, self).build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    position_embeddings = self._position_embeddings[:input_shape[1], :]
    return tf.broadcast_to(position_embeddings, input_shape)


