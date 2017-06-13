# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from nets.vgg16 import vgg16
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import scipy.sparse
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb

def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Fast R-CNN network."""
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    with sess.Graph().as_default():
      tf.set_random_seed(cfg.RNG_SEED)
      net = vgg16(batch_size=100)
      cls_prob = net.build_network()

      sess.run(tf.global_variables_initializer()) # initialize the variables
      feed_dict = {net._image:roidb[0],net.label:roidb[1]}
      cls_prob = sess.run(cls_prob,feed_dict=feed_dict)

      print(cls_prob)
  print('done solving')
