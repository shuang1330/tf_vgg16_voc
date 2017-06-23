# read the txt file
# f = open('all_tensors.txt', 'r')
# names = []
#
# for line in f:
#     if line.startswith('tensor_name'):
#         names.append(line)
#
#
# names = sorted(names)
# for name in names:
#     print name
#
#
# f.close()
from __future__ import absolute_import
import _init_paths
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

all_tensors = True

path = '/volume/home/shuang/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default'
name = 'vgg16_faster_rcnn_iter_25000.ckpt'
file_path = os.path.join(path,name)
reader = pywrap_tensorflow.NewCheckpointReader(file_path)

if all_tensors:
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print key
        print reader.get_tensor(key).shape
