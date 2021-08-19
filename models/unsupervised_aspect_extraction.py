#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:


import tensorflow as tf
import tensorflow.keras as keras


class AvgEmbedding(keras.layers.Layer):
  def call(self, inputs):
    if len(inputs) == 2:
      emb, mask = inputs
    else:
      emb = inputs[0]
      mask = None

    # shape: batch * hidden_size
    if mask:
      sum_emb = tf.reduce_sum(emb, axis=-2)
      avg_emb = sum_emb / tf.reduce_sum(mask, axis=-1)
    else:
      avg_emb = tf.reduce_mean(emb, axis=-2)
    return avg_emb


class Attention(keras.layers.Layer):
  def __init__(self, hidden_size, seq_len):
    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.M = keras.layers.Dense(self.hidden_size, use_bias=False,
                                kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                name='M')

  def call(self, inputs, **kwargs):
    if len(inputs) == 3:
      word_emb, sentence_emb, mask = inputs
    else:
      word_emb, sentence_emb = inputs
      mask = None

    # word_emb * M
    # batch * seq_len * hidden_size
    atten = self.M(word_emb)
    # tile avg_emb, shape: batch * seq * hidden_size
    avg_emb_tile = tf.expand_dims(sentence_emb, axis=-2)
    avg_emb_tile = tf.tile(avg_emb_tile, [1, self.seq_len, 1])
    # atten * avg_emb_tile, shape: batch * seq * hidden_size
    atten_score = atten * avg_emb_tile
    # shape: batch * seq
    atten_score = tf.reduce_sum(atten_score, axis=-1)

    atten_score = keras.layers.Softmax()(atten_score, mask)
    return atten_score


class UnsupervisedAspectExtraction(keras.models.Model):
  def __init__(self, hidden_size, num_cluster, seq_len, vocab_size, word_emb_init):
    super(UnsupervisedAspectExtraction, self).__init__()

    self.hidden_size = hidden_size
    self.num_cluster = num_cluster
    self.seq_len = seq_len

    self._avg_emb = AvgEmbedding()
    self._attention = Attention(hidden_size, seq_len)

    self.word_emb = keras.layers.Embedding(vocab_size, hidden_size, weights=[word_emb_init],
                                           trainable=False, name='word_embedding')
    self.pt_dense = keras.layers.Dense(num_cluster, activation='softmax', name='pt_dense')
    self.rs_dense = keras.layers.Dense(hidden_size, use_bias=False, name="rs_dense")

  def get_config(self):
    pass

  def call(self, inputs, training=None, mask=None):
    pos_word_ids, pos_mask, neg_word_ids, neg_mask = inputs
    pos_word_emb = self.word_emb(pos_word_ids)
    neg_word_emb = self.word_emb(neg_word_ids)

    # pos_sent_embedding, ys in the paper, shape: batch * hidden
    pos_sent_emb = tf.reduce_mean(pos_word_emb, axis=-2)
    # attention
    atten_score = self._attention([pos_word_emb, pos_sent_emb, pos_mask])
    # Zs, shape: batch * hidden
    zs = tf.reduce_sum(pos_word_emb * tf.expand_dims(atten_score, axis=-1), axis=-2)
    # Pt, shape: batch * num_cluster
    pt = self.pt_dense(zs)
    # Rs, shape: batch * hidden
    rs = self.rs_dense(pt)

    # neg_sent_embedding, zn in the paper, shape: batch * hidden
    neg_sent_emb = tf.reduce_mean(neg_word_emb, axis=-2)

    # loss
    rs_zs = tf.matmul(tf.expand_dims(rs, axis=1), tf.expand_dims(zs, axis=-1))
    rs_zn = tf.matmul(tf.expand_dims(rs, axis=1), tf.expand_dims(neg_sent_emb, axis=-1))
    loss = tf.math.maximum(0, 1 - rs_zs + rs_zn)

    return loss
