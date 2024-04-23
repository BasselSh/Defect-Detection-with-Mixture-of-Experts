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
from wrappers import *
# model_path = "configs/swin/swin_tiny.py"
model_path = "configs/swin/swin_moe.py"
ckpt_path = "weights/epoch_24.pth"
cfg = Config.fromfile(model_path)
model = init_detector(cfg)
# model = MODELS.build(cfg.model)

vis = DetLocalVisualizer()
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (0, 200, 30), (100, 200, 150), (100, 200, 50), (10, 200, 50)]
CLASSES = ('1','2','3','4','5','6')
img = mmcv.imread("data/mmdet/NEU_DET/test/inclusion_251.jpg")

from mmengine.runner.checkpoint import _load_checkpoint
checkpoint = _load_checkpoint(ckpt_path, 'cpu')
ckp_state_dict = checkpoint['state_dict']
model_state_dict = model.state_dict()

experts = []
for i, key in enumerate(model_state_dict.keys()):
    if 'experts' in key:
        experts.append(key)

ffn_ckpt = []
for i, key in enumerate(ckp_state_dict.keys()):
    if 'ffn' in key:
        ffn_ckpt.append(key)

num_experts = 4
j = 1
ckp_it = iter(ckp_state_dict)
model_it = iter(model_state_dict)
experts_it = iter(experts)
l = 219
it = 0
stop_it = False
while not stop_it:
    model_key = next(model_it)
    # print("MODEL")
    # print(model_key)
    if 'gate' in model_key:
        model_state_dict[model_key] = torch.nn.Parameter(model_state_dict[model_key])
        print("GATE")
        print(model_state_dict[model_key])
        # model_state_dict[model_key] = torch.rand(model_state_dict[model_key].shape)
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
            shape = model_state_dict[model_key].shape
            # shape = [shape[0], shape[1]] if len(shape)==2 else [shape[0]]
            # print(shape)
            # print(ws[i%4].shape)
            model_state_dict[model_key] = ws[i%4].clone()
            i+=1
            model_key = next(model_it)  
        if 'gate' in model_key:
            model_state_dict[model_key] = torch.nn.Parameter(model_state_dict[model_key])
            print("GATE")
            print(model_state_dict[model_key].shape)
            model_key = next(model_it)
            model_state_dict[model_key] = torch.nn.Parameter(model_state_dict[model_key])
            print("GATE")
            print(model_state_dict[model_key].shape)
            continue
    ckp_key = next(ckp_it)
    # print(ckp_key)
    it+=1
    if it==l:
            stop_it = True
            break
    model_state_dict[model_key] = ckp_state_dict[ckp_key].clone()
    
revise_keys=[(r'^module\.', '')]
from mmengine.runner.checkpoint import _load_checkpoint_to_model
out = _load_checkpoint_to_model(model, model_state_dict, revise_keys)
results = inference_detector(model, img)
# # print(results.keys())
# results.pred_instances.labels = torch.ones_like(results.pred_instances.labels)
# results.pred_instances = results.pred_instances.detach().cpu()
# output = vis._draw_instances(img, results.pred_instances[results.pred_instances.scores>0.01], classes=CLASSES, palette=palette)
# # plt.imshow(output)