from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __init__paths
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from model.train import train_net
from dataset.read_roidb import pascal_voc
import numpy as np
import sys
import cv2
import math

import tensorflow as tf
from nets.vgg16 import vgg16
import random

SHUFFLE = True

def one_hot(number):
  lis = np.zeros([21,])
  lis[number] = 1
  return lis

def load_batch(db,citer,batch_size,roidb):
    data,label =[],[]
    end_index = (citer+1)*batch_size

    if citer == 0:
        roidb = random.shuffle(roidb)
        # return something

    if end_index > db.num_images and SHUFFLE:
        # get images till num_images and start from the 0 index again
        end_index = db.num_images
        rest_images = batch_size-end_index+start
        data,label = split_roidb(roidb,citer,end_index)
        # shuffle the roidb
        roidb = random.shuffle(roidb)

    for roi in roidb[citer*batch_size:end_index]:
        path = os.path.join(db.imagepath,roi['index']+'.jpg')
        [x1,y1,x,y] = roi['box']
        print(x1,x,y1,y)
        data.append(cv2.resize(cv2.imread(path)[y1:y,x1:x,:],(224,224)))
        label.append(roi['gt_class'])
    data = np.asarray(data)
    label = np.asarray(label)
    return data,label

if __name__ == '__main__':
    db = pascal_voc()
    trainval_roidb = db.read_roidb('trainval')
    test_roidb = db.read_roidb('test')
    batch_size = 50
    max_iters = 1000

    with tf.Session().as_default() as sess:
        with sess.graph.as_default() as g:
            # build the graph
            images = tf.placeholder(tf.float32,shape=[batch_size,
                                    224, 224, 3])
            labels = tf.placeholder(tf.int32,shape=[batch_size,])
            cls_score, cls_prob = vgg16(images,batch_size)
            print(cls_score.shape)
            print(labels.shape)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(cls_score, [-1, db.num_classes]),
                labels=labels))
            saver = tf.train.Saver()
            #training settings
            lr = tf.Variable(0.001, trainable=False)
            momentum = tf.Variable(0.9, trainable=False)
            optimizer = tf.train.MomentumOptimizer(lr,momentum)
            gvs = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gvs)
            #initialize the network
            iter_in_this_epoch = 0
            epoch = 0
            citer = math.ceil(epoch/batch_size) + iter_in_this_epoch
            print('loading %dth batch of images'%(citer))
            train_x, train_y = load_batch(db,citer,batch_size,trainval_roidb)
            print('loaded')
            sess.run(tf.global_variables_initializer())#snapshots!
            #start training
            feed_dict = {images:train_x,labels:train_y}
            while citer< max_iters+1:
                cls_score,cls_prob,loss,_ = sess.run([cls_score,
                                        cls_prob,
                                        loss,
                                        train_op],
                                        feed_dict=feed_dict)
                # display training info
                if citer % 100 == 0:
                    print('iter:%d/%d\n>>>loss:%.6f, lr:%f'%
                            (citer,max_iters,loss,lr.eval()))
                #snapshots
                if citers % 1000 == 0:
                    filename = os.path.join()
                    saver.save(sess.filename)

                citer += 1
