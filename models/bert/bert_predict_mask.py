#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import json
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from mytf2.encoder import BertEncoder
from mytf2.tokenizer import FullTokenizer
from mytf2.layer.activation import get_activation
from mytf2.loader import checkpoint_loader, load_bert_weights_from_official_checkpoint


class BiasLayer(keras.layers.Layer):
    def __init__(self, size, name):
        super(BiasLayer, self).__init__()

        self.bias = self.add_weight(name, [size], initializer='zeros')

    def call(self, x):
        return x + self.bias


class BERT(keras.Model):
    def __init__(self, bert_config):
        super(BERT, self).__init__()
        self._bert_config = bert_config
        self.bert = BertEncoder(**bert_config)
        with tf.name_scope("cls/predictions"):
            with tf.name_scope("transform"):
                self.transform = keras.layers.Dense(bert_config["hidden_size"],
                                                    activation=get_activation(bert_config["hidden_act"]))
                self.transform_norm = keras.layers.LayerNormalization()

            self.output_bias = BiasLayer(bert_config["vocab_size"], "output_bias")

    def call(self, inputs, training=False):
        seq_output = self.bert(inputs)[0]
        last_layer_output = seq_output[-1]
        reps = self.transform(last_layer_output)
        reps = self.transform_norm(reps)

        logits = tf.matmul(reps, self.bert.get_embedding_table(), transpose_b=True)
        logits = self.output_bias(logits)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        return log_probs

    def get_config(self):
        return {
            "bert_config": self._bert_config
        }

    def init_weight(self, bert_init_ckpt):
        # load bert encoder weight
        load_bert_weights_from_official_checkpoint(self.bert, self._bert_config["max_position_embeddings"],
                                                   bert_init_ckpt)
        # load prediction dnn weight
        loader = checkpoint_loader(bert_init_ckpt)
        self.transform.set_weights([
            loader("cls/predictions/transform/dense/kernel"),
            loader("cls/predictions/transform/dense/bias")
        ])
        self.transform_norm.set_weights([
            loader("cls/predictions/transform/LayerNorm/gamma"),
            loader("cls/predictions/transform/LayerNorm/beta")
        ])
        self.output_bias.set_weights([
            loader("cls/predictions/output_bias")
        ])


def sent_preprocess(tokenizer, text):
    text_list = text.split("[MASK]")
    len_list = len(text_list)
    tokens = ["[CLS]"]
    for i in range(len_list):
        sub_text = text_list[i]
        if sub_text != "":
            tokens.extend(tokenizer.tokenize(sub_text))
        if i != len_list - 1:
            tokens.append("[MASK]")

    return tokens


def build_input(tokenizer, text, max_len):
    tokens = sent_preprocess(tokenizer, text)
    valid_len = len(tokens)
    if valid_len > max_len:
        tokens = tokens[:max_len]
        valid_len = max_len
    else:
        tokens.extend(["[PAD]"] * (max_len - valid_len))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * valid_len + [0] * (max_len - valid_len)
    segment_ids = [0] * max_len
    input_ids = np.array(input_ids, dtype=np.int32).reshape([1, max_len])
    input_mask = np.array(input_mask, dtype=np.int32).reshape([1, max_len])
    segment_ids = np.array(segment_ids, dtype=np.int32).reshape([1, max_len])
    return input_ids, input_mask, segment_ids


def replace_mask(text):
    star_cnt = text.count("*")
    mask_idx = []
    while star_cnt:
        mask_idx.append(text.index("*"))
        text = text.replace("*", "å", 1)
        star_cnt -= 1

    text = text.replace("å", "[MASK]")
    return text, mask_idx


def main():
    bert_dir = "/Users/Damcy/Desktop/clinicalNLP/bert"
    bert_config_file = os.path.join(bert_dir, "bert_config.json")
    vocab_file = os.path.join(bert_dir, "vocab.txt")
    bert_init_ckpt = os.path.join(bert_dir, "bert_model.ckpt")

    bert_config = json.load(open(bert_config_file, 'r'))
    max_seq_len = bert_config["max_position_embeddings"]

    tokenizer = FullTokenizer(vocab_file)
    model = BERT(bert_config)

    fake_input = np.ones([max_seq_len], dtype=np.int32)
    model.predict([fake_input, fake_input, fake_input])
    model.init_weight(bert_init_ckpt)

    content = input("请输入句子，[MASK]可使用*代替：")
    while content != "exit":
        content, mask_idx = replace_mask(content)
        print(content)
        input_ids, input_mask, segment_ids = build_input(tokenizer, content, max_seq_len)
        log_probs = model.predict([input_ids, input_mask, segment_ids])
        for idx in mask_idx:
            idx += 1
            vocab_score = log_probs[0][idx]
            vocab_score = [[score, i] for i, score in enumerate(vocab_score)]
            vocab_score = sorted(vocab_score, key=lambda x: x[0], reverse=True)
            # top 10
            top10_ids = [x[1] for x in vocab_score[:10]]
            top10_score = [x[0] for x in vocab_score[:10]]
            top10_words = tokenizer.convert_ids_to_tokens(top10_ids)
            res = ["{}({:.2f})".format(w, s) for w, s in list(zip(top10_words, top10_score))]
            res = " ".join(res)
            print("位置{}预测结果：{}".format(idx, res))

        content = input("请输入句子，[MASK]可使用*代替：")


if __name__ == "__main__":
    main()
