# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg


class vgg16(batch_size=500):
  def __init__(self):
    self._batch_size = batch_size
    self._image = tf.placeholder(tf.float32,
    shape=[self._batch_size, None, None, 3])

  def build_network(self, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16',
                           regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      with tf.variable_scope('conv1'):
          net = slim.conv2d(self._image, self._filter_num[0],
          [3, 3], trainable=False, scope='conv1_1')  # 64
          net = slim.conv2d(net, self._filter_num[1],
          [3, 3], trainable=False, scope='conv1_2')  # 64
      net = slim.max_pool2d(net, [2,2], padding = 'SAME', scope='pool1')
      with tf.variable_scope('conv2'):
          net = slim.conv2d(net, self._filter_num[2],
          [3, 3], trainable = False, scope='conv2_1')  # 128
          net = slim.conv2d(net, self._filter_num[3],
          [3, 3], trainable = False, scope='conv2_2')  # 128
      net = slim.max_pool2d(net, [2,2], padding = 'SAME', scope = 'pool2')
      with tf.variable_scope('conv3'):
          net = slim.conv2d(net, self._filter_num[4],
          [3, 3], trainable = is_training, scope='conv3_1')  # 256
          net = slim.conv2d(net, self._filter_num[5],
          [3, 3], trainable = is_training, scope='conv3_2')  # 256
          net = slim.conv2d(net, self._filter_num[6],
          [3, 3], trainable = is_training, scope='conv3_3')  # 256
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      with tf.variable_scope('conv4'):
          net = slim.conv2d(net, self._filter_num[7],
          [3, 3], trainable = is_training, scope='conv4_1')  # 512
          net = slim.conv2d(net, self._filter_num[8],
          [3, 3], trainable = is_training, scope='conv4_2')  # 512
          net = slim.conv2d(net, self._filter_num[9],
          [3, 3], trainable = is_training, scope='conv4_3')  # 512
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      with tf.variable_scope('conv5'):
          net = slim.conv2d(net, self._filter_num[10],
          [3, 3], trainable = is_training, scope='conv5_1')  # 512
          net = slim.conv2d(net, self._filter_num[11],
          [3, 3], trainable = is_training, scope='conv5_2')  # 512
          net = slim.conv2d(net, self._filter_num[12],
          [3, 3], trainable = is_training, scope='conv5_3')  # 512
      net = slim.max_pool2d(net, [2,2], padding='SAME', scope='pool5')
      pool5_flat = slim.flatten(net, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, scope='dropout7')
      cls_score = slim.fully_connected(fc7, self._num_classes,
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
      cls_prob = tf.nn.softmax(cls_score, name="cls_prob")

      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob
      self._score_summaries.update(self._predictions)

      return cls_prob
