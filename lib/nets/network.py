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

# from tensorflow.python.client import timeline
# from tf_cnnvis import *

import numpy as np
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_target_layer import proposal_target_layer

from model.config import cfg


class Network(object):
  def __init__(self, batch_size=1):
    self._batch_size = batch_size
    self._predictions = {}
    self._losses = {}

    self._labels = [] # try to find the labels
    self._layers = {}
    self._score_summaries = {} # contain self._predictions
    self._train_summaries = [] # contain all trainable variables
    self._event_summaries = {} # contain self._losses

    self._filter_num = None

  def _add_image_summary(self, image, boxes):
    # add back mean
    image += cfg.PIXEL_MEANS
    # bgr to rgb (opencv uses bgr)
    channels = tf.unstack (image, axis=-1)
    image    = tf.stack ([channels[2], channels[1], channels[0]], axis=-1)
    # dims for normalization
    width  = tf.to_float(tf.shape(image)[2])
    height = tf.to_float(tf.shape(image)[1])
    # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
    cols = tf.unstack(boxes, axis=1)
    boxes = tf.stack([cols[1] / height,
                      cols[0] / width,
                      cols[3] / height,
                      cols[2] / width], axis=1)
    # add batch dimension (assume batch_size==1)
    assert image.get_shape()[0] == 1
    boxes = tf.expand_dims(boxes, dim=0)
    image = tf.image.draw_bounding_boxes(image, boxes)

    return tf.summary.image('ground_truth', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
      tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def build_network(self, sess, is_training=True):
    raise NotImplementedErrors

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('loss_' + self._tag) as scope:
      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._labels, [-1]) #!

      cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

      self._losses['cross_entropy'] = cross_entropy
      loss = cross_entropy
      self._losses['total_loss'] = loss

      self._event_summaries.update(self._losses)

    return loss

  def create_architecture(self, sess, mode, num_classes,
        tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2),
        filter_num = (64,64,128,128,256,256,256,512,512,512,512,512,512,512)):
    self._image = tf.placeholder(tf.float32,
    shape=[self._batch_size, None, None, 3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag
    self._filter_num = filter_num


    self._num_classes = num_classes
    self._mode = mode

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = None
    else:
      biases_regularizer = tf.no_regularizer

    with arg_scope([slim.conv2d, slim.fully_connected],
                    biases_regularizer=biases_regularizer,
                    biases_initializer=tf.constant_initializer(0.0)):
      cls_prob = self.build_network(sess, training)

    layers_to_output = {}
    layers_to_output.update(self._predictions)

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if mode == 'TRAIN':
      self._add_losses()
      layers_to_output.update(self._losses)

    val_summaries = []
    with tf.device("/cpu:0"):
      val_summaries.append(self._add_image_summary(self._image, self._gt_boxes))
      for key, var in self._event_summaries.items():
        val_summaries.append(tf.summary.scalar(key, var))
      for key, var in self._score_summaries.items():
        self._add_score_summary(key, var)
      for var in self._train_summaries:
        self._add_train_summary(var)

    self._summary_op = tf.summary.merge_all()
    if not testing:
      self._summary_op_val = tf.summary.merge(val_summaries)

    return layers_to_output

  # only useful during testing mode
  def extract_conv5(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["conv5_3"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob = sess.run([self._predictions["cls_score"],
                                    self._predictions['cls_prob']],
                                    feed_dict=feed_dict)

    return cls_score, cls_prob, bbox_pred, rois

  # only useful during testing mode
  def test_image_with_timeline(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    cls_score, cls_prob, = sess.run([self._predictions["cls_score"],
                                     self._predictions['cls_prob']],
                                     options = run_options, #
                                     run_metadata = run_metadata, #
                                     feed_dict=feed_dict)

    return cls_score, cls_prob

  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info']}
    loss, cls_score, cls_prob, _ = sess.run([self._losses["rpn_cross_entropy"],
                                            self._losses['total_loss'],
                                            self._predictions["cls_score"],
                                            self._predictions["cls_prob"],
                                            train_op],
                                            feed_dict=feed_dict)
    return loss_cls, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info']}
    loss_cls, loss, summary, _ = sess.run([self._losses['cross_entropy'],
                                            self._losses['total_loss'],
                                            self._summary_op,
                                            train_op],
                                            feed_dict=feed_dict)
    return loss_cls, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info']}
    sess.run([train_op], feed_dict=feed_dict)
