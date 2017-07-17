from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __init__paths
from path import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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

def activations_decay_generator(x):
    return tf.nn.sigmoid(x)(1-tf.nn.sigmoid(x))

if __name__ == '__main__':
    # rcnn_models = PATH_FASTER_RCNN_MODEL
    rcnn_models = '../faster_rcnn_models/vgg16_faster_rcnn_iter_70000.ckpt'
    outputdir = OUTPUT_DIR
    tbdir = '../tensorboard'
    db = pascal_voc()
    trainval_roidb = db.read_roidb('trainval')
    # print(db.num_images)
    testdb = pascal_voc()
    test_roidb = testdb.read_roidb('test')
    # print(db.num_images)
    # print(testdb.num_images)
    batch_size = 128
    max_iters = 10000
    # # define the rank of activations
    # act_sum = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as sess:
        with sess.graph.as_default() as g:
            # build the graph
            images = tf.placeholder(tf.float32,shape=[None,
                                    224, 224, 3])
            labels = tf.placeholder(tf.int32,shape=[None,])
            cls_score, cls_prob, act_summaries = vgg16(images,batch_size,ACT=True)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(cls_score, [-1, db.num_classes]),
                labels=labels))

            # # added loss for activations
            # loss_act = tf.Variable(0,name='LOSS_ACT')
            # for index,act in enumerate(act_summaries):
            #     decay = activations_decay_generator(act_sum[index])
            #     loss_act += decay*tf.nn.l1_loss(act)

            tf.summary.histogram('SCORE/loss',loss)
            var_to_restore = []
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                # print(var.name)
                if 'weights' in var.name:
                    var_to_restore.append(var)
                #     print(var.name)
                # if 'var.name'.endswith('biases'):
                    # print(var.name)
            # raise NotImplemented
                # tf.summary.histogram('TRAIN/'+var.op.name,var) # summaries
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver(var_list=var_to_restore)
            #training settings
            lr_1 = tf.Variable(0.0005, trainable=False) # 0.001
            momentum_1 = tf.Variable(0.01, trainable=False) # 0.9
            optimizer_1 = tf.train.MomentumOptimizer(lr_1,momentum_1)
            gvs = optimizer_1.compute_gradients(loss)
            train_op = optimizer_1.apply_gradients(gvs)
            #initialize the network
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, rcnn_models) # restore from frcnn models
            # start training
            citer = math.ceil(epoch/batch_size) + iter_in_this_epoch
            # write summaries
            writer = tf.summary.FileWriter(tbdir, g)
            while citer< max_iters+1:
                # print('loading %dth batch of images'%(citer))
                train_x, train_y = load_batch(db,epoch,iter_in_this_epoch,
                    batch_size,trainval_roidb)
                # print('loaded')
                feed_dict = {images:train_x,labels:train_y}
                if citer%50 != 0:
                    score,prob,total_loss, _ = sess.run([cls_score,
                                            cls_prob,
                                            loss,
                                            train_op],
                                            feed_dict=feed_dict)
                # display training info
                if citer % 50 == 0:
                    score,prob,total_loss, summaries, _ = \
                                    sess.run([cls_score,
                                            cls_prob,
                                            loss,
                                            summary_op,
                                            train_op],
                                            feed_dict=feed_dict)
                    # write summaries
                    # writer.add_summary(summaries, float(citer))
                    print('>>>iter:%d/%d, epoch:%d, lr:%f\n  training loss:%.6f'
                        %(citer,max_iters,epoch,lr_1.eval(),total_loss))
                # snapshots
                if citer % 50 == 0:
                    # ckpt_prefix = 'vgg_voc_%s'%int(citer)
                    # filename = os.path.join(outputdir,ckpt_prefix+'.ckpt')
                    # saver.save(sess,filename)
                    # print('saved snapshot in %s'%filename)
                    # display test info
                    test_x, test_y = load_batch(testdb,
                                        0,0,128,test_roidb)
                    feed_dict_test = {images:test_x,labels:test_y}
                    test_loss= sess.run(loss,feed_dict=feed_dict_test)
                    print('  test loss: %.4f'%test_loss)
                citer += 1
            writer.close()
