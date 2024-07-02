from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils.dl_utils import tensor2imgs
import cv2
import numpy as np
import os.path as osp
import mmcv

@HOOKS.register_module()
class ImageVisualizer(Hook):
    def __init__(self,
                interval: int = 1,
                draw_gt: bool = True,
                draw_pred: bool = True,
                num_images=2,
                iterations=10):
        super().__init__()
        self.num_images = num_images
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self._interval = interval
        self.iterations = iterations
    
    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self._interval):
            data_sample = outputs[0]
            inputs= data_batch['inputs']
            input = inputs[0].permute(1,2,0).numpy()
            ori_shape = data_sample.ori_shape
            origin_image = cv2.resize(input, ori_shape)
            name = osp.basename(data_sample.img_path)
            runner.visualizer.add_datasample(name, origin_image,
                                                data_sample,
                                                self.draw_gt, self.draw_pred, step=self.iterations)



