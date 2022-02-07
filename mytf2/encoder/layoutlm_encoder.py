#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 


import tensorflow as tf

from mytf2.layer.embedding import OnDeviceEmbedding
from mytf2.layer.attention import SelfAttentionMask
from mytf2.layer.transformer import TransformerEncoder
from mytf2.layer.activation import get_activation, serialize_activation


@tf.keras.utils.register_keras_serializable(package="mytf2")
class LayoutLMEncoder(tf.keras.layers.Layer):
  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               max_sequence_length=512,
               max_position_embeddings=1000,
               intermediate_size=3072,
               activation="gelu",
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
               **kwargs):
    super(LayoutLMEncoder, self).__init__(**kwargs)
    self.activation = get_activation(activation)
    self.initializer = tf.keras.initializers.get(initializer)

    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_attention_heads = num_attention_heads
    self.max_sequence_length = max_sequence_length
    self.max_position_embeddings = max_position_embeddings
    self.intermediate_size = intermediate_size
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate

  def get_config(self):
    config = {
      'vocab_size': self.vocab_size,
      'hidden_size': self.hidden_size,
      'num_layers': self.num_layers,
      'num_attention_heads': self.num_attention_heads,
      'max_sequence_length': self.max_sequence_length,
      'max_position_embeddings': self.max_position_embeddings,
      'intermediate_size': self.intermediate_size,
      'activation': serialize_activation(self.activation),
      'dropout_rate': self.dropout_rate,
      'attention_dropout_rate': self.attention_dropout_rate,
      'initializer': tf.keras.initializers.serialize(self.initializer),
    }
    base_config = super(LayoutLMEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    embedding_width = self.hidden_size
    with tf.name_scope("embeddings"):
      self._word_embedding_layer = OnDeviceEmbedding(
        vocab_size=self.vocab_size,
        embedding_width=embedding_width,
        initializer=self.initializer,
        name='word_embeddings'
      )

      self._x_position_embedding_layer = OnDeviceEmbedding(
        vocab_size=self.max_position_embeddings,
        embedding_width=embedding_width,
        initializer=self.initializer,
        name="x_position_embeddings"
      )

      self._y_position_embedding_layer = OnDeviceEmbedding(
        vocab_size=self.max_position_embeddings,
        embedding_width=embedding_width,
        initializer=self.initializer,
        name="y_position_embeddings"
      )

      self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

      self._embedding_dropout_layer = tf.keras.layers.Dropout(
        name='dropout', rate=self.dropout_rate)

    self._transformer_layers = []
    for i in range(self.num_layers):
      layer = TransformerEncoder(
        num_attention_heads=self.num_attention_heads,
        intermediate_dim=self.intermediate_size,
        intermediate_activation=self.activation,
        kernel_initializer=self.initializer,
        output_dropout=self.dropout_rate,
        attention_dropout=self.attention_dropout_rate,
        name='transformer/layer_%d' % i
      )
      self._transformer_layers.append(layer)

  def call(self, inputs, training=None):
    word_ids, mask, xy_positions = inputs
    x0, y0, x1, y1 = xy_positions
    word_embeddings = self._word_embedding_layer(word_ids)
    # position embeddings
    x0_embeddings = self._x_position_embedding_layer(x0)
    x1_embeddings = self._x_position_embedding_layer(x1)
    y0_embeddings = self._y_position_embedding_layer(y0)
    y1_embeddings = self._y_position_embedding_layer(y1)
    embeddings = tf.keras.layers.Add()(
      [word_embeddings, x0_embeddings, x1_embeddings, y0_embeddings, y1_embeddings]
    )

    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout_layer(embeddings, training)

    attention_mask = SelfAttentionMask()(embeddings, mask)

    encoder_outputs = []
    data = embeddings
    for layer in self._transformer_layers:
        data = layer([data, attention_mask])
        encoder_outputs.append(data)

    return encoder_outputs
