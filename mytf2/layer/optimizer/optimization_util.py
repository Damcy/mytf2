#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 


import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers

from .warm_up import WarmUp
from .adam_weight_decay_optimizer import AdamWeightDecay


def create_optimizer(init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     end_lr=0.0,
                     optimizer_type='adamw'):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps,
      end_learning_rate=end_lr)
  if num_warmup_steps:
    lr_schedule = WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=lr_schedule,
        warmup_steps=num_warmup_steps)

  if optimizer_type == 'adamw':
    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
  elif optimizer_type == 'lamb':
    optimizer = tfa_optimizers.LAMB(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
  else:
    raise ValueError('Unsupported optimizer type: ', optimizer_type)

  return optimizer

