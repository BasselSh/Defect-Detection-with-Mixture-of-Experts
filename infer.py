import mmengine
import mmcv
from mmcv.transforms import Compose, Resize
import matplotlib.pyplot as plt
from mmdet.utils import get_test_pipeline_cfg
import torch
import numpy as np
import torchvision.transforms as torch_trans
from mmdet.apis import init_detector, inference_detector 
from mmdet.visualization import DetLocalVisualizer
import cv2
import glob
import os
from mmengine.structures import InstanceData

cfg_path = "configs/swin/swin_tiny.py"
ckpt_path = "weights/3cls.pth"
model = init_detector(cfg_path, ckpt_path)
vis = DetLocalVisualizer()

palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (0, 200, 30), (100, 200, 150), (100, 200, 50), (10, 200, 50)]
CLASSES = ('1','2','3','4','5','6')

root = '/home/huemorgen/quadrotor-plan-control/images_from_camera'
# root = '/home/huemorgen/projects/cam'
imgs_paths = os.listdir(root)

for img_path in imgs_paths:
    img = mmcv.imread(f"{root}/{img_path}")
    results = inference_detector(model, img)
    results.pred_instances = results.pred_instances.detach().cpu()
    bboxes = results.pred_instances.bboxes
    # ids = []
    # for i in range(bboxes.shape[0]):
    #     x1,y1,x2,y2 = bboxes[i]
    #     mask = torch.bitwise_and(torch.bitwise_and(x1 < bboxes[:,0], y1 < bboxes[:,1]), torch.bitwise_and(x2 > bboxes[:,2], y2 > bboxes[:,3]))
    #     if mask.any():
    #         continue
    #     ids.append(i)
    # if len(ids) != 0:
    #     pred_instances = InstanceData(metainfo=dict(img_shape=(800, 1196, 3)))
    #     pred_instances.bboxes = results.pred_instances.bboxes[ids]
    #     pred_instances.labels = results.pred_instances.labels[ids]
    #     pred_instances.scores = results.pred_instances.scores[ids]
    #     results.pred_instances = pred_instances
    
    output = vis._draw_instances(img, results.pred_instances[results.pred_instances.scores>0.7], classes=CLASSES, palette=palette)
    cv2.imwrite(f"cam_out_scratches/{img_path}",output)