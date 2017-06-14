from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __init__paths
from path import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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


if __name__ == '__main__':
    pre_models = '../output/vgg_voc_1000.0.ckpt'
    outputdir = OUTPUT_DIR
    db = pascal_voc()
    test_roidb = db.read_roidb('test')
    print('%d images in the test dataset'%db.num_images)
    max_iters = 10000
    batch_size = 1

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as sess:
        with sess.graph.as_default() as g:
            # build the graph
            images = tf.placeholder(tf.float32,shape=[batch_size,
                                    224, 224, 3])
            labels = tf.placeholder(tf.int32,shape=[batch_size,])
            cls_score,cls_prob,act_summaries = vgg16(images,batch_size,True)
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in variables:
                tf.summary.histogram('TRAIN/'+var.op.name,var)
            saver = tf.train.Saver()
            #load the weights
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, pre_models)
            #test the nets
            tp = 0.0
            for i,roi in enumerate(test_roidb):
                pa = os.path.join(db.imagepath,roi['index']+'.jpg')
                [x1,y1,x,y] = roi['box']
                test_x = [cv2.resize(cv2.imread(pa)[y1:y,x1:x,:],(224,224))]
                test_x = np.asarray(test_x)
                test_y = np.asarray([roi['gt_class']])
                feed_dict = {images:test_x,labels:test_y}
                prob,acts = sess.run([cls_prob,
                                    act_summaries],
                                    feed_dict=feed_dict)
                # naive way of counting correct results
                if np.argmax(prob[0])==test_y[0]:
                    tp += 1.0
                if i%50 == 0 and i>0:
                    print(tp/i)
                    print('processing %dth image'%(i))

                # write acts
                act_file = open('../acts/%s.txt'%i,'w')
                act_file.write('%s\n'%str(test_y[0]))
                act_file.write('\n')
                sum_act = []
                for arr in acts:
                    temp = np.sum(arr,axis = (0,1,2))
                    sum_act.append(temp)
                for item in sum_act:
                    act_file.write('{}\n'.format(str(item)))
                act_file.close()

            accuracy = tp/(i+1)
            print('accuracy for test dataset: %.2f'%accuracy)
