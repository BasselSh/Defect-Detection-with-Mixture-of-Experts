import json
import numpy as np
import os.path as osp
from pathlib import Path

def filter_classes(annotations_path, output_path, desired_cats):
    ann_pth = annotations_path
    data = json.load(open(ann_pth))
    anns = data['annotations']
    images = data['images']
    cats = data['categories']

    #Extract ids of the desired categories
    cat_ids_t = []
    cats_list = []
    for i, cat in enumerate(cats):
         if cat['name'] in desired_cats:
              cat_ids_t.append(cat['id'])
              cats_list.append(cat)

    #Filter annotations of the desired categories    
    image_ids = set()
    anns_list = []
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id in cat_ids_t:
            anns_list.append(ann)
            image_ids.add(ann['image_id'])
    
    #Filter images whose annotations are the of desired categories
    images_list = []
    for img in images:
        if img['id'] in image_ids:
            images_list.append(img)
    
    output_file = dict(images=images_list, annotations=anns_list, categories=cats_list)
    with open(output_path, 'w') as f:
            json.dump(output_file, f,indent=4, separators=(',', ': '))

if __name__=="__main__":
    annotations_path = "/home/huemorgen/Defect-Detection-with-Mixture-of-Experts/data/mmdet/NEU_DET/annotations/instances_train.json"
    output_path = "/home/huemorgen/Defect-Detection-with-Mixture-of-Experts/data/mmdet/NEU_DET/annotations/3classes_train.json"

    desired_cats = ['inclusion','patches', 'scratches']
    filter_classes(annotations_path, output_path, desired_cats)