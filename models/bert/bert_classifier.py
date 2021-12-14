#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:

import argparse
import json
import os

import tensorflow as tf
import tensorflow.keras as keras

from mytf2.encoder import BertEncoder
from mytf2.layer.optimizer import create_optimizer
from mytf2.loader import load_bert_weights_from_official_checkpoint


LOG = tf.get_logger()
LOG.setLevel('INFO')


def parse_option():
  parse = argparse.ArgumentParser()
  parse.add_argument("--do_train", dest="do_train", help="do train or not", default=False, action="store_true")
  parse.add_argument("--do_eval", dest="do_eval", help="do eval or not", default=False, action="store_true")
  parse.add_argument("--do_test", dest="do_test", help="do test or not", default=False, action="store_true")

  parse.add_argument("--vocab_file", dest="vocab_file", help="model vocab file", type=str)
  parse.add_argument("--config_file", dest="config_file", help="model config file", type=str)
  parse.add_argument("--bert_init_ckp", dest="bert_init_ckp", help="pre-trained bert model weight",
                     default=None, type=str)
  parse.add_argument("--model_init_ckp", dest="model_init_ckp", help="trained model weight",
                     default=None, type=str)

  parse.add_argument("--max_seq_len", dest="max_seq_len", help="max sequence length", default=128, type=int)
  parse.add_argument("--num_epoch", dest="num_epoch", help="num of epoch", default=1, type=int)
  parse.add_argument("--learning_rate", dest="learning_rate", help="learning rate", default=3e-5, type=float)
  parse.add_argument("--save_ckp_steps", dest="save_ckp_steps", help="save checkpoint (per) steps",
                     default=int(1e4), type=int)
  parse.add_argument("--data_dir", dest="data_dir", help="data dir", type=str)
  parse.add_argument("--train_batch_size", dest="train_batch_size", help="train batch size", default=1, type=int)
  parse.add_argument("--eval_batch_size", dest="eval_batch_size", help="eval batch size", default=1, type=int)
  parse.add_argument("--test_batch_size", dest="test_batch_size", help="test batch size", default=1, type=int)

  parse.add_argument("--output_dir", dest="output_dir", help="output dir", type=str)

  (option, args) = parse.parse_known_args()
  return option, args


def build_model(bert_config, num_class, learning_rate,
                num_train_steps=None, bert_init_ckp=None):
  sequence_length = bert_config["max_seq_len"]
  dropout_rate = bert_config["dropout_rate"]
  hidden_size = bert_config["hidden_size"]

  # input
  input_ids = keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
  input_mask = keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_mask")
  segment_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="segment_ids")

  # bert model
  bert = BertEncoder(**bert_config)
  seq_output, _ = bert([input_ids, input_mask, segment_ids])
  # cls
  last_layer_output = seq_output[-1]
  cls_output = keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1), last_layer_output)
  cls_output = keras.layers.Dropout(rate=dropout_rate)(cls_output)

  # cls classifier
  hidden = keras.layers.Dense(hidden_size // 2,
                              kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(cls_output)
  logit = keras.layers.Dense(num_class, kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                             activation="softmax", name="label_ids")(hidden)

  model = keras.Model(inputs=[input_ids, input_mask, segment_ids],
                      outputs=[logit])

  if num_train_steps:
    optimizer = create_optimizer(learning_rate, num_train_steps=num_train_steps,
                                 num_warmup_steps=int(num_train_steps * 0.1),
                                 optimizer_type="adamw")
  else:
    optimizer = keras.optimizers.Adam(learning_rate)

  model.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy")

  if bert_init_ckp:
    LOG.info("load bert init ckp from: %s", bert_init_ckp)
    load_bert_weights_from_official_checkpoint(bert, bert_config, bert_init_ckp,
                                               load_pooler_layer=False)

  return model


def get_dataset_batch(tf_record_file, seq_len, batch_size,
                      shuffle=True, shuffle_buf=10000, num_parse_threads=2,
                      drop_remainder=True, repeat=True):
  feature = {
    "input_ids": tf.io.FixedLenFeature([seq_len], tf.int32),
    "input_mask": tf.io.FixedLenFeature([seq_len], tf.int32),
    "segment_ids": tf.io.FixedLenFeature([seq_len], tf.int32),
    "label_ids": tf.io.FixedLenFeature([batch_size], tf.int32)
  }

  def decode_record(example_string):
    example = tf.io.parse_single_example(example_string, feature)
    example = ({"input_ids": example["input_ids"], "input_mask": example["input_mask"],
                "segment_ids": example["segment_ids"]},
               {"label_ids": example["label_ids"]})
    return example

  dataset = tf.data.TFRecordDataset(tf_record_file)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=shuffle_buf)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.map(decode_record, num_parallel_calls=num_parse_threads)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
  return dataset


def get_dataset_len(tf_record_file):
  dataset = tf.data.TFRecordDataset(tf_record_file)
  count = dataset.reduce(0, lambda x, _: x + 1).numpy()
  return count


def train(model, train_data, num_epoch, steps_per_epoch, output_dir):
  callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "ckp-loss{loss:.6f}")),
    keras.callbacks.TensorBoard(os.path.join(output_dir, "logs"),
                                histogram_freq=0, embeddings_freq=0,
                                update_freq=1)
  ]

  model.fit(train_data, verbose=1, epochs=num_epoch, steps_per_epoch=steps_per_epoch,
            callbacks=callbacks)
  # save final ckp
  final_ckp_file = os.path.join(output_dir, "ckp-final")
  model.save(final_ckp_file)


def test(model, test_data, output_dir):
  output_file = os.path.join(output_dir, "label_test.txt")

  with open(output_file, 'w') as f:
    for (x_data, y_data) in test_data:
      logits = model.predict(x_data)
      num_samples = logits.shape[0]
      for i in range(num_samples):
        pred_id = keras.backend.argmax(logits[i], axis=-1).numpy()
        f.write(str(pred_id) + "\n")


def main():
  config, _ = parse_option()

  bert_config = json.load(open(config.config_file, 'r'))

  data_dir = config.data_dir

  train_data = None
  num_train_steps = None
  num_train_steps_per_epoch = None
  if config.do_train:
    train_file = os.path.join(data_dir, "train.tf_record")
    num_train_example = get_dataset_len(train_file)
    num_train_steps_per_epoch = num_train_example // config.train_batch_size
    num_train_steps = num_train_steps_per_epoch * config.num_epoch

    LOG.info("num of training example: %d", num_train_example)
    LOG.info("num of training steps per epoch: %d", num_train_steps_per_epoch)
    LOG.info("num of total training steps: %d", num_train_steps)

    train_data = get_dataset_batch(train_file, seq_len=config.max_seq_len,
                                   batch_size=config.train_batch_size)

  if config.model_init_ckp:
    LOG.info("load model from: %s", config.model_init_ckp)
    model = keras.models.load_model(config.model_init_ckp)
  else:
    model = build_model(bert_config, config.num_class, config.learning_rate,
                        num_train_steps=num_train_steps, bert_init_ckp=config.bert_init_ckp)

  if config.do_train:
    train(model, train_data,
          num_epoch=config.num_epoch, steps_per_epoch=num_train_steps_per_epoch,
          output_dir=config.output_dir)

  if config.do_test:
    test_file = os.path.join(data_dir, "test.tf_record")
    test_data = get_dataset_batch(test_file, seq_len=config.max_seq_len,
                                  batch_size=config.test_batch_size)
    test(model, test_data, config.output_dir)


if __name__ == '__main__':
  main()
