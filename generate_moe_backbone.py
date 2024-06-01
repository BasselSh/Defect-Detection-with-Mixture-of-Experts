import mmengine
import mmcv
from mmcv.transforms import Compose, Resize
import matplotlib.pyplot as plt
from mmdet.utils import get_test_pipeline_cfg
import torch
import numpy as np
import torchvision.transforms as torch_trans
from mmdet.apis import init_detector, inference_detector 
from wrappers import *
from mmdet.visualization import DetLocalVisualizer
from mmengine.registry import MODELS
from mmengine.config import Config
from mmengine.runner.checkpoint import save_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate MOE Backbone')
    parser.add_argument('--moe-cfg', default="configs/swin/swin_moe.py", help='path to MOE config') 
    args = parser.parse_args()
    return args


def main():
    '''
    This script produces backbone weights to be loaded in moe-based swin backbone. The procedure is as follow:
    1. Load checkpoints of swin from detectron, and save them in mmdet format.
    2. Reload the checkpoints which are in mmdet format, and insert the weights in the state_dict of the swin-moe backbone.
    3. Ouptut the swin-moe backbone for later use.

    '''
    #initializing paths
    args = parse_args()
    moe_path = args.moe_cfg
    swin_path = "configs/swin/swin_tiny.py"
    ckpt_path = "weights/swin_tiny_patch4_window7_224.pth"
    swin_file = 'swin_base_model.pth'
    moe_file = 'moe_base_model.pth'

    ''' 1. converting the detectron swin weights into mmdet'''
    cfg = Config.fromfile(swin_path)
    model = init_detector(cfg, ckpt_path)
    save_checkpoint(model.backbone.state_dict(), swin_file) #save the backbone weights

    ''' 2.1 Reloading the mmdet weights, and extract the state_dict'''
    ckpt_path = swin_file
    checkpoint = _load_checkpoint(ckpt_path, 'cpu')
    ckp_state_dict = checkpoint

    ''' 2.2 Loading the moe model'''
    cfg = Config.fromfile(moe_path)
    num_experts = cfg.model.backbone.num_experts #get the number of experts
    model = init_detector(cfg) 
    backbone = model.backbone #get only the backbone
    backbone_state_dict = backbone.state_dict() #get the randomly initialized backbone state_dict of swin-moe

    ''' 2.3 get module names containing gates'''
    gates = []
    for i, key in enumerate(backbone_state_dict.keys()):
        if 'gate' in key:
            gates.append(key)

    ''' 2.4 get module names of experts'''
    experts = []
    for i, key in enumerate(backbone_state_dict.keys()):
        if 'experts' in key:
            experts.append(key)

    ''' 2.5 get module names of ffn layers (used only for sanity check)'''
    ffn_ckpt = []
    for i, key in enumerate(ckp_state_dict.keys()):
        if 'ffn' in key:
            ffn_ckpt.append(key)


    load_checkpoint_to_moe(ckp_state_dict, backbone_state_dict, num_experts)

    checkpoint_out = dict(state_dict=backbone_state_dict)
    save_checkpoint(checkpoint_out, moe_file)



def load_checkpoint_to_moe(ckp_state_dict, backbone_state_dict, num_experts):
    j = 1
    ckp_it = iter(ckp_state_dict)
    model_it = iter(backbone_state_dict)

    l = len([key for key in ckp_state_dict])
    it = 0
    stop_it = False
    while not stop_it:
        model_key = next(model_it)
        print("Module")
        print(model_key)
        if 'gate' in model_key:
            continue
    
        if 'expert' in model_key:
            ws = []
            for i in range(4):
                key = next(ckp_it)
                w = ckp_state_dict[key]
                ws.append(w)
                it+=1
                if it==l:
                    stop_it = True
                    break
            i = 0
            for _ in range(4*num_experts):
                shape = backbone_state_dict[model_key].shape
                # shape = [shape[0], shape[1]] if len(shape)==2 else [shape[0]]
                print(shape)
                print(ws[i%4].shape)
                backbone_state_dict[model_key] = ws[i%4].clone()
                i+=1
                model_key = next(model_it)  
            if 'gate' in model_key:
                model_key = next(model_it)
                continue
        ckp_key = next(ckp_it)
        print(ckp_key)
        it+=1
        if it==l:
                stop_it = True
                break
        backbone_state_dict[model_key] = ckp_state_dict[ckp_key].clone()


if __name__=="__main__":
    main()