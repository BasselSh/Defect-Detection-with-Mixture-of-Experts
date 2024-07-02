import json
import numpy as np
import os.path as osp
from pathlib import Path

def _split_images(images,threshold):
    splitted_ids=dict(train=[], test=[])
    id2img = dict()
    for img in images:
        file_name = img['file_name']
        file_name += ".jpg" if file_name[-4:]!='.jpg' else ''
        img['file_name'] = file_name
        filename_splitted = file_name.split("_")
        num = int(filename_splitted[-1].split(".")[0])
        id = img['id']
        id2img[id] = img
        mode = "train" if num<=threshold else "test"
        splitted_ids[mode].append(id)
    return splitted_ids, id2img

def split_annotation_file(annotations_file, test_size, train_ouput_name, test_output_name):
    ''' 
    Function to split a json coco-annotated file into train and test files based on the number of images of every class.
    Files names of images must be written in the following format: DEFECTNAME_NUMBEROFIMAGE.jpg for example patches_230.jpg
    '''
    ann_pth = Path(annotations_file)
    output_name = dict(train=train_ouput_name, test=test_output_name)

    with open(ann_pth, 'r') as file:
        data = json.load(file)
    
    anns = data['annotations']
    images = data['images']
    cats = data['categories']

    threshold = int(len(images)/len(cats)*(1-test_size)) # number of images to keep for training. 
                                                         #In NEU-DET dataset there are 300 images of every class.
                                                         #Wtih test_size=1/5, there will be 240 training images.
    splitted_ids, id2img = _split_images(images, threshold)

    train_test = ('train', 'test')
    for mode in train_test:
        ids = splitted_ids[mode]
        np.random.shuffle(ids)
        output_file = dict( images=[id2img[id] for id in ids],
                            annotations=[ann for ann in anns if ann['image_id'] in ids],
                            categories=cats)
        
        name = output_name[mode]
        annotations_dir = ann_pth.parent
        save_pth = osp.join(annotations_dir, name+".json")
        print(save_pth)
        with open(save_pth, 'w') as f:
            json.dump(output_file, f,indent=4, separators=(',', ': '))

if __name__=="__main__":
    ##parameters##
    annotations_file = "/home/huemorgen/Defect-Detection-with-Mixture-of-Experts/data/mmdet/NEU_DET/annotations/all_samples.json"
    test_size = 1/5
    train_ouput_name = "instances_train2"
    test_output_name = "instances_test2"

    split_annotation_file(annotations_file, test_size, train_ouput_name, test_output_name)
    
    
            
        
    
    
    
        
    