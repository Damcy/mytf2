#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 


import tensorflow as tf
import tensorflow.keras as keras

from mytf2.encoder import BertEncoder
from mytf2.layer import CRF


class SeqBert(keras.Model):
  def __init__(self, model_config, label_num):
    super(SeqBert, self).__init__()
    self.bert = BertEncoder(**model_config)
    self.dropout = keras.layers.Dropout(model_config['hidden_dropout_prob'])
    self.dense = keras.layers.Dense(label_num, kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                    activation="softmax", name="logits")
    self.crf = CRF(label_num)

  def call(self, inputs, training=False):
    input_ids, input_mask, segment_ids, label_ids = inputs
    seq_output = self.bert([input_ids, input_mask, segment_ids])[0]
    last_layer_output = seq_output[-1]
    if training:
      last_layer_output = self.dropout(last_layer_output, training=training)
    logits = self.dense(last_layer_output)
    seq_lens = tf.reduce_sum(input_mask, -1)
    log_likelihood, pred_ids = self.crf([logits, seq_lens, label_ids])
    loss = tf.reduce_mean(-log_likelihood)
    return loss, logits, pred_ids

  def get_bert_module(self):
    return self.bert
