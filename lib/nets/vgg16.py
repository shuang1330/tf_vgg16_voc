from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

num_classes = 21

def vgg16(images, batch_size, ACT=False, is_training=True,
            filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512]):
    act_summaries = []
    with tf.variable_scope('vgg_16', 'vgg_16',
        regularizer=tf.contrib.layers.l2_regularizer(0.5)): # 0.0005
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope('conv1'):
            net = slim.conv2d(images, filter_num[0],
                [3, 3], trainable=False, scope='conv1_1')  # 64
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[1],
                [3, 3], trainable=False, scope='conv1_2')  # 64
            act_summaries.append(net)
        net = slim.max_pool2d(net, [2,2], padding = 'SAME', scope='pool1')
        with tf.variable_scope('conv2'):
            net = slim.conv2d(net, filter_num[2],
                [3, 3], trainable = False, scope='conv2_1')  # 128
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[3],
                [3, 3], trainable = False, scope='conv2_2')  # 128
            act_summaries.append(net)
        net = slim.max_pool2d(net, [2,2], padding = 'SAME', scope = 'pool2')
        with tf.variable_scope('conv3'):
            net = slim.conv2d(net, filter_num[4],
            [3, 3], trainable = False, scope='conv3_1')  # 256
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[5],
            [3, 3], trainable = False, scope='conv3_2')  # 256
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[6],
            [3, 3], trainable = False, scope='conv3_3')  # 256
            act_summaries.append(net)
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
        with tf.variable_scope('conv4'):
            net = slim.conv2d(net, filter_num[7],
            [3, 3], trainable = False, scope='conv4_1')  # 512
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[8],
            [3, 3], trainable = False, scope='conv4_2')  # 512
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[9],
            [3, 3], trainable = False, scope='conv4_3')  # 512
            act_summaries.append(net)
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
        with tf.variable_scope('conv5'):
            net = slim.conv2d(net, filter_num[10],
            [3, 3], trainable = False, scope='conv5_1')  # 512
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[11],
            [3, 3], trainable = False, scope='conv5_2')  # 512
            act_summaries.append(net)
            net = slim.conv2d(net, filter_num[12],
            [3, 3], trainable = False, scope='conv5_3')  # 512
            act_summaries.append(net)
        net = slim.max_pool2d(net, [2,2], padding='SAME', scope='pool5')
        [a,b,c,d] = net.get_shape().as_list()
        pool5_flat = slim.flatten(net, [batch_size,b*c*d], scope='flatten')
        fc6 = slim.fully_connected(pool5_flat,4096,trainable=is_training,scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, scope='dropout6')
        fc7 = slim.fully_connected(fc6,4096,trainable=is_training,scope='fc7')
        if is_training:
            fc7 = slim.dropout(fc7, scope='dropout7')
        cls_score = slim.fully_connected(fc7, num_classes,
                                    weights_initializer=initializer,
                                    trainable=is_training,
                                    activation_fn=None, scope='cls_score')
        cls_prob = tf.nn.softmax(cls_score, name="cls_prob")


        # predictions["cls_score"] = cls_score
        # predictions["cls_prob"] = cls_prob
        # score_summaries.update(predictions)

    if ACT:
        return cls_score, cls_prob, act_summaries
    return cls_score, cls_prob
