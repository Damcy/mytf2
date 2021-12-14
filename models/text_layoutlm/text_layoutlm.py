#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:

import tensorflow as tf
import tensorflow.keras as keras

from mytf2.encoder import LayoutLMEncoder


class LayoutLM(keras.Model):
  def __init__(self, model_config):
    super(LayoutLM, self).__init__()
    self.model_config = model_config
    # encoder
    self.layoutLMEncoder = LayoutLMEncoder(**model_config)
    # mlm dnn
    initializer = keras.initializers.TruncatedNormal(stddev=0.02)
    with tf.name_scope("cls/prediction"):
      self.hidden = keras.layers.Dense(model_config["hidden_size"],
                                       kernel_initializer=initializer,
                                       activation=model_config["activation"], name="transform")
      self.layer_norm = keras.layers.LayerNormalization()
      self.dense = keras.layers.Dense(model_config["vocab_size"],
                                      kernel_initializer=initializer,
                                      activation="softmax", name="output")

  def call(self, inputs, training=None, mask=None):
    input_ids, input_mask, x0, y0, x1, y1 = inputs
    seq_output = self.layoutLMEncoder([input_ids, input_mask, [x0, y0, x1, y1]])
    rep = seq_output[-1]
    rep = self.hidden(rep)
    rep = self.layer_norm(rep)
    rep = self.dense(rep)
    return rep

  def get_config(self):
    return {"model_config": self.model_config}
