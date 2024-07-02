from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
from typing import Dict, List, Optional, Sequence, Union
import torch 
from mmdet.evaluation.functional import bbox_overlaps
import numpy as np

@METRICS.register_module()
class GenericMetric(CocoMetric):
    
    default_prefix = "NEU_DET"
    
    def __init__(self,
                 dataset_name: Optional[str] = "NEU_DET",
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 *args, **kwargs) -> None:

        super().__init__(ann_file=ann_file,
                        metric=metric,
                        *args, **kwargs)
        self.default_prefix = dataset_name

@METRICS.register_module()
class SegMetric(CocoMetric):
    
    default_prefix = "NEU_DET"
    
    def __init__(self,
                 dataset_name: Optional[str] = "NEU_DET",
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 num_classes = 6,
                 *args, **kwargs) -> None:

        super().__init__(ann_file=ann_file,
                        metric=metric,
                        *args, **kwargs)
        self.default_prefix = dataset_name
        self.iou_thresh = 0.5
        self.TPs = torch.zeros(num_classes)
        self.Gts = torch.zeros(num_classes)
        self.area = torch.zeros(num_classes)
        self.num_classes = num_classes

    def process(self, data_batch, data_samples):
        for output in data_samples:
            segmap_shape = output['ori_shape']

            pred = output['pred_instances']
            bboxes_pred = pred['bboxes']
            scores_pred  = pred['scores']
            labels_pred  = pred['labels'].type(torch.int)

            gts = output['gt_instances']
            bboxes_gt = gts['bboxes']
            labels_gt = gts['labels'].type(torch.int)
            bboxes_gt_all = dict()
            for i in range(self.num_classes):
                bboxes_gt_by_label = bboxes_gt[labels_gt==i].reshape(-1,4)
                if len(bboxes_gt_by_label)==0:
                    continue

                # print(bboxes_gt_by_label)
                self.Ad_ious = torch.tensor(bbox_overlaps(bboxes_gt_by_label.detach().cpu().numpy(), 
                                                          bboxes_gt_by_label.detach().cpu().numpy())) - torch.eye(bboxes_gt_by_label.shape[0])

                bboxes_gt_united = self._unit_bboxes(bboxes_gt_by_label)
                bboxes_gt_all[i] = bboxes_gt_united

            assert len(bboxes_gt_all) <= bboxes_gt.shape[0],\
                'somthing is wrong'
            
            for k in range(self.num_classes):
                if not k in bboxes_gt_all:
                    continue
                bboxes_gt_one_class = bboxes_gt_all[k]
                segmap_pred = self._bboxes2segmap(bboxes_pred[labels_pred==k], one_label=k, segmap_shape=segmap_shape)
                for i, bbox_gt in enumerate(bboxes_gt_one_class):
                    bbox_label = labels_gt[i]
                    if isinstance(bbox_gt, list):
                        segmap_gt = self._bboxes2segmap(bbox_gt, one_label=bbox_label, segmap_shape=segmap_shape)
                    else:
                        segmap_gt = self._single_bbox2segmap(bbox_gt, bbox_label, segmap_shape)

                    tp, area_covered = self._tp_single_pred(segmap_pred, segmap_gt) #should add score

                    self.TPs += tp
                    self.Gts[bbox_label] += 1
                    self.area += area_covered     

    def _single_bbox2segmap(self,bbox, label, segmap_shape=None):
        assert segmap_shape is not None,\
            'undefined segmentation shape'
        
        W, H = segmap_shape
        segmap_gt = torch.zeros((6,H,W), dtype=torch.int)
        x1, y1, x2, y2 = self._extract_bbox(bbox)
        segmap_gt[label, y1:y2, x1:x2] = 1
        return segmap_gt
    
    def _bboxes2segmap(self, bboxes, labels=None, segmap_shape=None, one_label=None):
        W, H = segmap_shape
        segmap = torch.zeros((6,H,W), dtype=torch.int)
        for i, bbox in enumerate(bboxes):
            if one_label is None:
                label = labels[i]
            else:
                label = one_label
            segmap = torch.bitwise_or(segmap, self._single_bbox2segmap(bbox, label, segmap_shape=segmap_shape))
        return segmap

    def _extract_bbox(self, bbox):
        x1, y1, x2, y2 = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
        return x1, y1, x2, y2

    def _tp_single_pred(self, segmap_pred, segmap_gt):
        area_covered = torch.sum(torch.bitwise_and(segmap_pred, segmap_gt), (1,2))
        gt_area = segmap_gt.sum((1,2))
        ratio_area_covered = torch.where(gt_area>0, area_covered/gt_area, 0)
        tp = torch.where(ratio_area_covered>self.iou_thresh, 1, 0)
        return tp, ratio_area_covered

    def _unit_bboxes(self, bboxes):
        all_boxes = []
        self.visited = set()
        for i in range(bboxes.shape[0]):
            s = set()
            self._search_neighbors(i, s)
            if len(s) == 0:
                continue
            if len(s) == 1:
                all_boxes.append(bboxes[i])
                continue
            ls = list(s)
            bboxes_united = [bboxes[j] for j in ls]
            all_boxes.append(bboxes_united)

        return all_boxes

    def _search_neighbors(self, i, s):
        if i in self.visited:
            return
        self.visited.add(i)
        s.add(i)
        Ad = self.Ad_ious
        neighbors = torch.nonzero(Ad[i])
        
        if len(neighbors) == 0:
            return
        
        for neigh in neighbors:
            self._search_neighbors(neigh.item(), s)

    def evaluate(self, size: int) -> Dict:
        metrics = dict()
        metrics['recall'] = self.TPs/self.Gts
        metrics['area'] = self.area/self.Gts
        if self.prefix:
            metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in metrics.items()
                }
        print(metrics)
        return metrics