#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:


import argparse
import json
import os

import numpy as np
import tensorflow as tf

from text_layoutlm import LayoutLM
from mytf2.layer.optimizer import create_optimizer
from mytf2.loader import load_bert_weights_from_official_checkpoint


def parse_option():
  parse = argparse.ArgumentParser()
  parse.add_argument("--train_file", dest="train_file", help="train file",
                     default=None, type=str)
  parse.add_argument("--output_dir", dest="output_dir", help="output dir",
                     default=None, type=str)
  parse.add_argument("--config_file", dest="config_file", help="model config file",
                     default=None, type=str)
  parse.add_argument("--bert_init_ckpt", dest="bert_init_ckpt", help="bert init ckpt",
                     default=None, type=str)
  parse.add_argument("--max_seq_len", dest="max_seq_len", help="max seq len",
                     default=512, type=int)
  parse.add_argument("--num_train_steps", dest="num_train_steps", help="num train steps",
                     default=10000, type=int)
  parse.add_argument("--save_model_freq", dest="save_model_freq", help="save model freq",
                     default=1000, type=int)
  parse.add_argument("--warm_up_steps", dest="warm_up_steps", help="num warm up steps",
                     default=1000, type=int)
  parse.add_argument("--train_batch_size", dest="train_batch_size", help="train batch size",
                     default=4096, type=int)
  parse.add_argument("--learning_rate", dest="learning_rate", help="learning rate",
                     default=2e-5, type=float)
  (option, args) = parse.parse_known_args()
  return option, args


# 载入数据集，构造训练batch，
def get_dataset_batch(tf_record_file, sequence_len, vocab_size, batch_size,
                      shuffle=True, repeat=True, drop_remainder=False,
                      shuffle_buffer_size=1000, num_parse_threads=4):
  MASK_ID = 103
  feature = {
    "input_ids": tf.io.FixedLenFeature([sequence_len], tf.int64),
    "input_mask": tf.io.FixedLenFeature([sequence_len], tf.int64),
    "input_x0": tf.io.FixedLenFeature([sequence_len], tf.int64),
    "input_y0": tf.io.FixedLenFeature([sequence_len], tf.int64),
    "input_x1": tf.io.FixedLenFeature([sequence_len], tf.int64),
    "input_y1": tf.io.FixedLenFeature([sequence_len], tf.int64)
  }

  def dynamic_maks(input_ids):
    real_seq_len = tf.reduce_sum(tf.sign(input_ids)).numpy()
    new_input_ids = input_ids.numpy()
    # 15% words
    num_replace_word = int(real_seq_len * 0.15)
    candidate_idx_list = np.random.choice(range(1, real_seq_len),
                                          num_replace_word, replace=False)
    # 80% replace by [MASK]
    pos_replace_mask = int(num_replace_word * 0.8)
    replace_mask_idxes = candidate_idx_list[:pos_replace_mask]
    for idx in replace_mask_idxes:
      new_input_ids[idx] = MASK_ID
    # 10% replace by random word
    pos_replace_random = int(num_replace_word * 0.9)
    replace_random_idxes = candidate_idx_list[pos_replace_mask:pos_replace_random]
    for idx in replace_random_idxes:
      new_input_ids[idx] = np.random.randint(1, vocab_size - 1)

    label_mask = [0] * sequence_len
    for idx in candidate_idx_list:
      label_mask[idx] = 1
    return new_input_ids, label_mask

  def decode_record(example_string):
    example = tf.io.parse_single_example(example_string, feature)
    # mlm preprocess: mask 15% tokens
    # 80% of them replace by [MASK], 10% replace by random tokens, 10% keep the same
    ori_input_ids = example["input_ids"]
    input_ids, label_mask = tf.py_function(dynamic_maks, inp=[ori_input_ids], Tout=[tf.int32, tf.int32])
    input_ids.set_shape([sequence_len])
    label_mask.set_shape([sequence_len])
    x = (input_ids, example["input_mask"], example["input_x0"], example["input_y0"],
         example["input_x1"], example["input_y1"])
    y = {"output_1": example["input_ids"]}
    return x, y, label_mask

  dataset = tf.data.TFRecordDataset(tf_record_file)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
  if repeat:
    dataset = dataset.repeat()

  dataset = dataset.map(decode_record, num_parallel_calls=num_parse_threads)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset


def main():
  arg, _ = parse_option()
  model_dir = os.path.join(arg.output_dir, "model")
  os.mkdir(model_dir)
  weight_dir = os.path.join(arg.output_dir, "weight")
  os.mkdir(weight_dir)

  model_config = json.load(open(arg.config_file, 'r'))
  model = LayoutLM(model_config)
  # predict before summary
  rand_input = np.random.randint(0, 2, arg.train_batch_size, arg.max_seq_len)
  model.predict((rand_input, rand_input, rand_input, rand_input, rand_input, rand_input))
  model.summary()

  # 定义优化器，从bert checkpoint初始化word embedding和Transformer参数
  optimizer = create_optimizer(arg.learning_rate, num_train_steps=arg.num_train_steps,
                               num_warmup_steps=arg.warm_up_steps, optimizer_type="lamb")
  model.compile(optimizer, loss={"output_1": "sparse_categorical_crossentropy"},
                metrics={"output_1": "sparse_categorical_accuracy"})
  if arg.bert_init_ckpt is not None and arg.bert_init_ckpt != "":
    print("load bert init ckpt from: %s".format(arg.bert_init_ckpt))
    load_bert_weights_from_official_checkpoint(model.layoutLMEncoder,
                                               arg.max_seq_len, arg.bert_init_ckpt,
                                               load_position_embedding=False,
                                               load_type_embedding=False,
                                               load_pooler_layer=False)

  # load training data 加载训练数据
  train_data = get_dataset_batch(arg.train_file, arg.max_seq_len, model_config["vocab_size"],
                                 arg.train_batch_size)
  # training
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, "text-layout-{loss:.5f}"),
                                       save_freq=arg.save_model_freq)
  ]
  model.fit(train_data, epochs=1, steps_per_epoch=arg.num_train_steps, callbacks=callbacks)

  # save the model
  final_ckpt_file = os.path.join(model_dir, "model.ckpt")
  model.save(final_ckpt_file)
  # save model's weight
  final_weight_file = os.path.join(weight_dir, "ckpt")
  model.save_weights(final_weight_file)


if __name__ == '__main__':
  main()
