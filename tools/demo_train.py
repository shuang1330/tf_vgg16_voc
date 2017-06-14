from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __init__paths
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from model.train import train_net
from nets.vgg16 import vgg16
from dataset.read_roidb import pascal_voc
import numpy as np
import sys
import cv2
import math

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


SHUFFLE = True
epoch = 0
iter_in_this_epoch = 0

def split_roidb(db,roidb,start,end):
    data,label = [],[]
    for roi in roidb[start:end]:
        path = os.path.join(db.imagepath,roi['index']+'.jpg')
        [x1,y1,x,y] = roi['box']
        data.append(cv2.resize(cv2.imread(path)[y1:y,x1:x,:],(224,224)))
        label.append(roi['gt_class'])
    data = np.asarray(data)
    label = np.asarray(label)
    return data,label

def load_batch(db,epoch,iter_in_this_epoch,batch_size,roidb):
    start = iter_in_this_epoch*batch_size
    end = (iter_in_this_epoch+1)*batch_size

    if iter_in_this_epoch==0 and batch_size==0 and SHUFFLE:
        roidb = random.shuffle(roidb)
        iter_in_this_epoch += 1
        return split_roidb(db,roidb,start,end)

    if end > db.num_images and SHUFFLE:
        # get images till num_images
        end = db.num_images
        data,label = split_roidb(db,roidb,start,end_index)
        # shuffle the roidb and start from the 0 index again
        roidb = random.shuffle(roidb)
        rest_images = batch_size-end+start
        rest_data,rest_label = split_roidb(db,roidb,0,rest_images)
        iter_in_this_epoch = 0
        epoch += 1
        return data+rest_data,label+rest_label

    else:
        iter_in_this_epoch += 1
        return split_roidb(db,roidb,start,end)

def get_variables_in_checkpoint_file(file_name):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")

if __name__ == '__main__':
    db = pascal_voc()
    trainval_roidb = db.read_roidb('trainval')
    test_roidb = db.read_roidb('test')
    batch_size = 50
    max_iters = 10000
    rcnn_models = \
     '../faster_rcnn_models/vgg16_faster_rcnn_iter_70000.ckpt'
    outputdir = '../output'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as sess:
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
            sess.run(tf.global_variables_initializer())#snapshots!
            # see all variables in the graph
            # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # see all variables in the ckpt file_index
            # ckpt_vars = get_variables_in_checkpoint_file(rcnn_models)
            # restorer = tf.train.Saver(variables)
            # for var in variables:
                # restorer.restore(sess,rcnn_models)
                # if var not in ckpt_vars:
                    # print(var.name)
            # raise NotImplementedError
            saver.restore(sess, rcnn_models) # restore from frcnn models
            # start training
            citer = math.ceil(epoch/batch_size) + iter_in_this_epoch
            while citer< max_iters+1:
                # print('loading %dth batch of images'%(citer))
                train_x, train_y = load_batch(db,epoch,iter_in_this_epoch,
                    batch_size,trainval_roidb)
                # print('loaded')
                feed_dict = {images:train_x,labels:train_y}
                score,prob,total_loss,_ = sess.run([cls_score,
                                        cls_prob,
                                        loss,
                                        train_op],
                                        feed_dict=feed_dict)
                # display training info
                if citer % 50 == 0:
                    print('iter:%d/%d, epoch:%d\n>>>loss:%.6f, lr:%f'%
                            (citer,max_iters,epoch,total_loss,lr.eval()))
                # snapshots
                if citer >0 and citer % 1000 == 0:
                    ckpt_prefix = 'vgg_voc_%s'%citer
                    filename = os.path.join(outputdir,ckpt_prefix+'.ckpt')
                    saver.save(sess,filename)
                    print('saved snapshot in %s'%filename)

                citer += 1
