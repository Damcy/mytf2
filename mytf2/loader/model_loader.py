#!/usr/bin/env python
# -*- encoding=utf-8 -*-
# author: ianma
# email: stmayue@gmail.com
# created time: 2021-04-09 17:39:04


import tensorflow as tf


def checkpoint_loader(checkpoint_file):
  def _loader(name):
    return tf.train.load_variable(checkpoint_file, name)
  return _loader


def load_bert_weights_from_official_checkpoint(block, config, checkpoint_file,
                                               load_word_embedding=True,
                                               load_position_embedding=True,
                                               load_type_embedding=True,
                                               load_pooler_layer=True):
  """Load trained official model from checkpoint.
  :param block: Built block.
  :param config: model configuration dict.
  :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
  :param load_word_embedding
  :param load_position_embedding
  :param load_type_embedding
  :param load_pooler_layer
  """
  loader = checkpoint_loader(checkpoint_file)

  if load_word_embedding:
    block._word_embedding_layer.set_weights([
      loader('bert/embeddings/word_embeddings'),
    ])

  if load_position_embedding:
    block._position_embedding_layer.set_weights([
      loader('bert/embeddings/position_embeddings')[:config['max_seq_len'], :],
    ])

  if load_type_embedding:
    block._type_embedding_layer.set_weights([
      loader('bert/embeddings/token_type_embeddings'),
    ])

  block._embedding_norm_layer.set_weights([
    loader('bert/embeddings/LayerNorm/gamma'),
    loader('bert/embeddings/LayerNorm/beta'),
  ])

  for i, _layer in enumerate(block._transformer_layers):
    _layer.set_weights([
      loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
      loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
      loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
      loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
      loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
      loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
      loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
      loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
      loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
      loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
      loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
      loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
      loader('bert/encoder/layer_%d/output/dense/kernel' % i),
      loader('bert/encoder/layer_%d/output/dense/bias' % i),
      loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
      loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
    ])

  if load_pooler_layer:
    block._pooler_layer.set_weights([
      loader('bert/pooler/dense/kernel'),
      loader('bert/pooler/dense/bias'),
    ])

