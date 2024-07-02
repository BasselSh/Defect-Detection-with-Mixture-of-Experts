import os.path as osp
import os
import numpy as np
import json
import copy
from sklearn.utils import shuffle

class MultiDatasetAnnotator(object):
    def __init__(self):
        self.start_id = 0
        self.ann_id = 0
        self.categories = False
        
    def combine_annotations(self, path1, path2, out_path):
        self.cat_id = 1
        data1 = json.load(open(path1))
        data2 = json.load(open(path2))
        keys = [data1, data2]
        datas_dict = dict(images=[], annotations=[], categories=[])
        for i,data in enumerate(keys):
            imgs = data['images']
            anns = data['annotations']
            cats = data['categories']
            if i==0:
                self.categories_len = len(cats)
            self.reset_ids(imgs, anns, cats, i)
            datas_dict['images'] +=imgs
            datas_dict['annotations'] +=anns
            datas_dict['categories'] +=cats
        
        datas_dict['images'] = shuffle(datas_dict['images'], random_state=1)
        datas_dict['annotations'] = shuffle(datas_dict['annotations'], random_state=1) 
        with open(out_path, 'w') as f:
            json.dump(datas_dict, f, indent=4, separators=(',', ': '))
        
    def reset_ids(self, imgs, anns, cats, change_cat_id):
        ids = len(imgs)
        
        id2newid = dict()
        id = self.start_id
        for img in imgs:
            cur_id = img['id']
            id2newid[cur_id] = id
            img['id'] = id
            id+=1
        for an in anns:
            newid = id2newid[an['image_id']]
            an['image_id'] = newid
            an['id'] = self.ann_id
            if change_cat_id!=0:
                an['category_id'] = an['category_id']+ self.categories_len
            self.ann_id+=1
        for cat in cats:
            cat['id'] = self.cat_id
            self.cat_id+=1
        self.start_id = ids

if __name__ == '__main__':
    anno = MultiDatasetAnnotator()
    trainfileout = 'data/train.json'
    testfileout = 'data/test.json'
    anno.combine_annotations('data/mmdet/NEU_DET/annotations/instances_train.json', 'data/oil_spill/annotations/instances_train.json', trainfileout)
    anno.combine_annotations('data/mmdet/NEU_DET/annotations/instances_test.json', 'data/oil_spill/annotations/instances_test.json', testfileout)