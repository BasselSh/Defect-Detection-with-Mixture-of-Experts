import glob
import os.path as osp
import os
import json
import argparse
import glob
from functools import partial

''' 
script to rename the images of a dataset
Enter the path to the annotation folder
The dataset directories and files should be as follow:

-dataset
    -annotations
        - *train*.json // * indicates any string. (can be null)
        - *val*.json
        - *test*.json (optional)
    -train
    -val
    -test (optional)
'''

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_folder', type=str, default=None, help='path to the annotation folder')
    parser.add_argument('--suffix', type=str, default='.jpg')
    return parser.parse_args()

def _rename_train_or_test(file, folder, parent, suffix='.jpg', mode='train'):
    if mode =='val':
        mode = 'valid'
    root = osp.join(parent, mode)
    img_dicts = file['images']
    for i, img_dict in enumerate(img_dicts):
        img_target = f'{mode}{i}'+suffix
        img_name = img_dict['file_name']
        src = osp.join(root, img_name)
        target = osp.join(root, img_target)
        os.rename(src, target)

        img_dict['file_name'] = img_target

    save_path = osp.join(folder, f'instances_{mode}.json')
    with open(save_path, 'w') as f:
        json.dump(file, f, indent=4, separators=(',', ': '))

def rename_dataset(ann_folder, suffix):
    parent = osp.dirname(ann_folder)
    ann_files = glob.glob(osp.join(ann_folder,'*'))
    rename_test_train = partial(_rename_train_or_test,\
                                folder=ann_folder, parent=parent, suffix=suffix)
    modes = ['train', 'val', 'test']
    while ann_files:
        ann_path = ann_files.pop(0)
        file = json.load(open(ann_path))
        for mode in modes:
            if mode in ann_path:
                rename_test_train(file, mode=mode)
                break

if __name__=="__main__":
    args = parse()
    suffix = args.suffix
    ann_folder = args.ann_folder
    rename_dataset(ann_folder, suffix)
    
    