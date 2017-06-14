import __init__paths
import xml.etree.cElementTree as ET
import os

class pascal_voc():
    def __init__(self):
        self.name = 'VOC2007'
        self.datapath = '../data/VOCdevkit2007/VOC2007'
        self.imagepath = '../data/VOCdevkit2007/VOC2007/JPEGImages'
        self.annpath = '../data/VOCdevkit2007/VOC2007/Annotations'
        self.classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(list(zip(self.classes,
                                list(range(self.num_classes)))))
        self.num_image = 0

    def load_annotations(self,index):
        ann_file = os.path.join(self.annpath,index+'.xml')
        tree = ET.parse(ann_file)
        objs = tree.findall('object')
        # Exclude the samples labeled as difficult
        non_diff_objs = [
          obj for obj in objs if int(obj.find('difficult').text) == 0]
        objs = non_diff_objs

        # Load object bounding boxes into a data frame.
        res = []
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self.class_to_ind[obj.find('name').text.lower().strip()]
            res.append({'index':index,'box':[x1,y1,x2,y2],'gt_class':cls})
        return res

    def read_roidb(self,db_type):
        db_path = os.path.join(self.datapath,'ImageSets/Main')
        f = open('{}/{}.txt'.format(db_path,db_type))
        file_index = [line[:-1] for line in f.readlines()]
        anns = [self.load_annotations(index) for index in file_index]
        anns = [j for i in anns for j in i] # [[a,b],[c]] -> [[a],[b],[c]]
        self.num_images = len(anns)
        return anns

if __name__=='__main__':
    d = pascal_voc()
    anns = d.read_roidb('trainval')
    print len(anns)
    print anns[0]
