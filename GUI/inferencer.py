import mmcv
from mmengine.registry import DATASETS
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmengine.runner import Runner
from mmdet.apis.inference import inference_detector, init_detector
from mmdet.visualization import DetLocalVisualizer
class Inferencer():
    def __init__(self, cfg, ckpt) -> None:
        self.model = init_detector(cfg, ckpt)
        self.visualizer = DetLocalVisualizer()
    
    def infer(self, img):
        datasample = inference_detector(self.model, img)
        self.visualizer.add_datasample(name='pred', image=img, data_sample=datasample, draw_gt=False, pred_score_thr=0.6)
        img = self.visualizer._image 
        return img


if __name__=='__main__':
    register_all_modules()