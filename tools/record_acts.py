import os.path
import matplotlib.pyplot as plt
import numpy as np

path = '../acts_NoTraining'
num_files = len([f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))])
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
arr_hm = [np.empty([num_files,21,64], dtype=float),
            np.empty([num_files,21,64], dtype=float),
            np.empty([num_files,21,128], dtype=float),
            np.empty([num_files,21,128], dtype=float),
            np.empty([num_files,21,256], dtype=float),
            np.empty([num_files,21,256], dtype=float),
            np.empty([num_files,21,256], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float)]
num_clas = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# calculate the activation versus classes matrix
print 'loading data from text files'
predictions = np.zeros([len(os.listdir(path)),21])
for file_ind,filename in enumerate(os.listdir(path)):
#     print 'processing file {}'.format(filename)
    clas = []
    acts = []
    f = open('/'.join([path,filename]),'r')
    act_ind = 0
    for line in f.readlines():
        if line and line[0].isdigit():
            clas.append(line[:-1])
        if line.startswith('['):
            if not line.endswith(']/n'):
                acts.append([])
                acts_this_line = line[2:-1].split(' ')
                for i in acts_this_line:
                    if i is not '':
                        acts[act_ind].append(float(i))
            else:
                raise IOError('Error line with fewer numbers than expected.')
        if line.startswith(' '):
            # print 'starts with nothing'
            if line.endswith(']\n'):
                acts_this_line = line[:-2].split(' ')
                for i in acts_this_line:
                    if i is not '':
                        acts[act_ind].append(float(i))
                act_ind += 1
            else:
                acts_this_line = line.split(' ')
                for i in acts_this_line:
                    if i is not '':
                        acts[act_ind].append(float(i))

    num_clas[int(clas[0])] += 1
    for j in range(13):
        arr_hm[j][file_ind][int(clas[0])] += acts[j]
print 'loaded'

# acts_vs_cls_recorder = open('../activations_res/res.txt','w')
acts_vs_cls_recorder = {}
for i in range(13):
    arr_hm_new = np.sum(arr_hm[i], axis=0)/num_clas[i]
    print arr_hm_new
    # save heatmap information to a dictionary
    acts_vs_cls_recorder['%dth_acts'%(i+1)] = arr_hm_new
save_path = './activations_res/res_NoTraining.npy'

np.save(save_path,acts_vs_cls_recorder)
print 'heatmap information saved in %s'%save_path
