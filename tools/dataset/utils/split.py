import json
import numpy as np
import os.path as osp

def split(images,threshold):
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
        
if __name__=="__main__":
    ##parameters##
    dataset_name = "NEU_DET"
    test_size = 1/5
    train_ouput_name = "instances_train"
    test_output_name = "instances_test"
    
    output_name = dict(train=train_ouput_name, test=test_output_name)

    cur_dirname = osp.dirname(__file__)
    ann_dir = osp.join(cur_dirname, f"../{dataset_name}/annotations")
    ann_pth = osp.join(ann_dir, "all_samples.json")

    with open(ann_pth, 'r') as file:
        data = json.load(file)
    
    anns = data['annotations']
    images = data['images']
    cats = data['categories']

    threshold = int(len(images)/len(cats)*(1-test_size))
    splitted_ids, id2img = split(images, threshold)

    train_test = ('train', 'test')
    for mode in train_test:
        ids = splitted_ids[mode]
        np.random.shuffle(ids)
        output_file = dict( images=[id2img[id] for id in ids],
                            annotations=[ann for ann in anns if ann['image_id'] in ids],
                            categories=cats)
        
        name = output_name[mode]
        save_pth = osp.join(ann_dir, name+".json")
        output_file = output_file
        with open(save_pth, 'w') as f:
            json.dump(output_file, f,indent=4, separators=(',', ': '))
    
            
        
    
    
    
        
    