#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import _init_paths
from model.config import cfg

from datasets.factory import get_imdb
from model.test import test_net

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import collections

from nets.vgg16 import vgg16
from tensorflow.python import pywrap_tensorflow

CLASSES = ('__background__','bus','bicycle','car','person', 'motorbike')
num_classes = 21
old_filter_num = (64,64,128,128,256,256,256,512,512,512,512,512,512,512)
new_filter_num = (64,64,128,128,256,256,256,512,512,512,128,512,512,512)
# name_scopes = ('vgg_16/conv1/conv1_1','vgg_16/conv1/conv1_2',
#                 'vgg_16/conv2/conv2_1','vgg_16/conv2/conv2_2',
#                 'vgg_16/conv3/conv3_1','vgg_16/conv3/conv3_2',
#                 'vgg_16/conv3/conv3_3','vgg_16/conv4/conv4_1',
#                 'vgg_16/conv4/conv4_2','vgg_16/conv4/conv4_3',
#                 'vgg_16/conv5/conv5_1','vgg_16/conv5/conv5_2',
#                 'vgg_16/conv5/conv5_3','vgg_16/rpn_conv/3x3')
names = ('weights','biases')
weights_path = None

def filter(dic):
    '''
    modify the weights_dic according the pruning rule(new_filter_num)
    inputs: collections.OrderedDict()
    '''
    biases = []
    weights = []
    w_momentum = []
    b_momentum = []
    name_scopes = []
    for name_scope in dic:
        name_scopes.append(name_scope)
        for name in dic[name_scope]:
            if name.startswith('weights'):
                # weights.append(dic[name_scope][name])
                weights.append(dic[name_scope][name]['value'])
                w_momentum.append(dic[name_scope][name]['Momentum'])
            elif name.startswith('biases'):
                # biases.append(dic[name_scope][name])
                biases.append(dic[name_scope][name]['value'])
                b_momentum.append(dic[name_scope][name]['Momentum'])

    diff = [(old_filter_num[ind] - new_filter_num[ind]) \
    for ind in range(len(old_filter_num))]

    current_ind = 0
    pre_ind = 0
    if diff[0] != 0:
        current_sum = np.sum(weights[0], axis = (0,1,2))
        current_ind = np.argsort(current_sum)
        weights[0] = np.delete(weights[0], current_ind[:diff[0]], axis = 3)
        biases[0] = np.delete(biases[0],current_ind[:diff[0]], axis = 0)
        if w_momentum[0] is not None:
            w_momentum[0] = np.delete(w_momentum[0], \
            current_ind[:diff[0]], axis = 3)
            b_momentum[ind] = np.delete(b_momentum[ind], \
            current_ind[:diff[0]], axis = 0)

    pre_ind = current_ind
    current_ind = None
    for ind in range(1,len(old_filter_num)):
        if diff[ind-1] != 0:
            weights[ind] = np.delete(weights[ind], pre_ind[:diff[ind-1]], \
            axis = 2)
            if diff[ind] == 0:
                pre_ind = None
        if diff[ind] != 0:
            current_sum = np.sum(weights[ind],axis = (0,1,2))
            current_ind = np.argsort(current_sum)
            weights[ind] = np.delete(weights[ind], current_ind[:diff[ind]], \
            axis = 3)
            biases[ind] = np.delete(biases[ind],current_ind[:diff[ind]], \
            axis = 0 )
            if w_momentum[ind] is not None:
                w_momentum[ind] = np.delete(w_momentum[ind], \
                current_ind[:diff[ind]], axis = 3)
                b_momentum[ind] = np.delete(b_momentum[ind], \
                current_ind[:diff[ind]], axis = 0)
            pre_ind = current_ind
            current_ind = None

    ind = 0
    for name_scope in dic:
        for name in dic[name_scope]:
            if ind <= len(old_filter_num):
                if name.startswith('weights'):
                    dic[name_scope][name]['value'] = weights[ind]
                    dic[name_scope][name]['Momentum'] = w_momentum[ind]
                elif name.startswith('biases'):
                    dic[name_scope][name]['value'] = biases[ind]
                    dic[name_scope][name]['Momentum'] = b_momentum[ind]
        ind += 1

    return dic

if __name__ == '__main__':
    '''
    load the old ckpt file and change the weights accordingly,
    then save it in a npy file
    '''
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    demonet = 'vgg16_faster_rcnn_iter_25000.ckpt'
    dataset = 'voc_2007_trainval'
    tfmodel = os.path.join('../output','vgg16',dataset, 'default', demonet)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError('{:s} not found'.format(tfmodel + '.meta'))

    reader = pywrap_tensorflow.NewCheckpointReader(tfmodel)

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True


    # load the network
    name_scopes = []
    with tf.Graph().as_default() as g1:
        with tf.Session(config=tfconfig, graph=g1).as_default() as sess:
            #load network
            net = vgg16(batch_size=1)
            net.create_architecture(sess,'TRAIN',num_classes,tag='default',
                                    anchor_scales=[8,16,32],
                                    filter_num=old_filter_num)
            saver = tf.train.Saver()
            saver.restore(sess,tfmodel)
            print 'Loaded network {:s}'.format(tfmodel)
            #get the weights from ckpt file
            dic = collections.OrderedDict()
            all_variables = []
            for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                name_list = item.name[:-2].split('/')
                name_scopes.append('/'.join(name_list[:-1]))

            for name_scope in name_scopes:
                for name in names:
                    tensor_name = '{}/{}'.format(name_scope,name)
                    momentum_name = '{}/Momentum'.format(tensor_name)
                    if name_scope not in dic:
                        try:
                            dic[name_scope] = \
                            {name:{'value':reader.get_tensor(tensor_name),\
                            'Momentum':reader.get_tensor(momentum_name)}}
                        except:
                            dic[name_scope] = \
                            {name:{'value':reader.get_tensor(tensor_name),\
                            'Momentum':None}}
                    else:
                        dic[name_scope][name] = \
                        {'value':reader.get_tensor(tensor_name),\
                        'Momentum':None}
                        try:
                            dic[name_scope][name]['Momentum'] = \
                            reader.get_tensor(momentum_name)
                        except:
                            continue

    # filter the weights
    dic = filter(dic)
    for name_scope in dic:
        for name in dic[name_scope]:
            try:
                print 'After filtering, the variable {}/{}/Momentum has shape {}'.format(
                name_scope, name, dic[name_scope][name]['Momentum'].shape)
            except:
                print 'After filtering, the variable {}/{}/Momentum is: '.format(
                name_scope, name, dic[name_scope][name]['Momentum'])

    # save the pruned weights to npy file
    folder_path = '../output/pruning/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    weights_name = 'pruned_conv11_momentum.npy'
    weights_path = os.path.join(folder_path,weights_name)
    np.save(weights_path, dic)
    print 'The weights are saved in {}'.format(weights_path)

    '''
    load the new weigts to a new graph,
    test the pruned network
    '''
    with tf.Graph().as_default() as g2:
        with tf.Session(config=tfconfig,graph=g2).as_default() as sess:
            #load the new graph
            net = vgg16(batch_size=1)
            net.create_architecture(sess,'TEST',num_classes,tag='default',
                                    anchor_scales = [8,16,32],
                                    filter_num = new_filter_num)

            # load the new weights from npy file
            weights_dic = np.load(weights_path).item()

            for name_scope in weights_dic:
                with tf.variable_scope(name_scope,reuse = True):
                    for name in weights_dic[name_scope]:
                        var = tf.get_variable(name)
                        sess.run(var.assign(weights_dic[name_scope][name]['value']))
                        print 'assign pretrain model to {}/{}'.format(name_scope,name)

            # test the new model
            imdb = get_imdb('voc_2007_test')
            filename = 'demo_pruning'
            test_net(sess, net, imdb, filename, max_per_image=100)
