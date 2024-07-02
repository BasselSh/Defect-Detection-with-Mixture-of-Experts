# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class NEUDETDataset(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes': ("crazing","inclusion","patches", "pitted_surface","rolled-in_scale","scratches"),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (0, 200, 30), (100, 200, 150)]
    }

@DATASETS.register_module()
class OilDataset(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes': ("oil"),
        'palette': [(220, 20, 60)]
    }