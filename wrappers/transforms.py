from mmdet.datasets.transforms import RandomAffine
import cv2

from mmdet.structures.bbox import autocast_box_type
from mmdet.registry import TRANSFORMS
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from numpy import random


@TRANSFORMS.register_module()
class RotateCenter(RandomAffine):

    def __init__(self, rotation_degree = None, withbbox=True, *args, **kwargs):
        self.rotation_degree = rotation_degree
        self.withbbox = withbbox
        super().__init__(*args, **kwargs)

    def _center_coordinates(self, height, width):
        trans_x = width/2
        trans_y = height/2
        translate_matrix_pre = self._get_translation_matrix(trans_x, trans_y)
        translate_matrix_post = self._get_translation_matrix(-trans_x, -trans_y)
        return translate_matrix_pre, translate_matrix_post

    
    @cache_randomness
    def _get_random_homography_matrix(self, height, width):
        
        # Rotation
        if self.rotation_degree is None:
            rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        else:
            rotation_degree = self.rotation_degree

        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        translate_matrix_pre, translate_matrix_post = self._center_coordinates(height, width)

        # Sandwishing transformations with translations
        warp_matrix = shear_matrix @ rotation_matrix @ scaling_matrix
        warp_matrix_centered = (
            translate_matrix_pre @ warp_matrix @ translate_matrix_post)
        
        return warp_matrix_centered

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape[:2]

        if self.withbbox:
            bboxes = results['gt_bboxes']
            num_bboxes = len(bboxes)
            if num_bboxes:
                bboxes.project_(warp_matrix)
                if self.bbox_clip_border:
                    bboxes.clip_([height, width])
                # remove outside bbox
                valid_index = bboxes.is_inside([height, width]).numpy()
                results['gt_bboxes'] = bboxes[valid_index]
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                    valid_index]
                results['gt_ignore_flags'] = results['gt_ignore_flags'][
                    valid_index]

                if 'gt_masks' in results:
                    raise NotImplementedError('RandomAffine only supports bbox.')
        return results