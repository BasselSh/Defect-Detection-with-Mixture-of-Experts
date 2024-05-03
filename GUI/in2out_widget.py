# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widgets.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from mmengine.runner import Runner
from imagecorruptions import corrupt
from mmcv import imread, imwrite
from PyQt5.QtGui import QPixmap, QImage
import mmcv
from mmengine.registry import DATASETS
from mmengine.config import Config
from mmdet.utils import register_all_modules
import copy
import mmengine
import os.path as osp
import json
import os
from widgets import UI_widgets
from in2out import Ui_Dialog
from wrappers.transforms import RotateCenter
import numpy as np
from corruptions import motion_blur
from mmdet.visualization import DetLocalVisualizer
from inferencer import Inferencer
# from imagecorruptions.corruptions import motion_blur
corruptions = [
            'none', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'motion_blur', 'zoom_blur', 'snow',
            'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise',
            'spatter', 'saturate', 'rotation'
        ]

# corruptions_dict = 

class AugmentGUI(Ui_Dialog):
    def setupUi(self, Dialog):
        super().setupUi(Dialog)
        self.categories = ['Crazing', 'Inclusion', 'Patches', 'Pitted-surface','Rolled-in scale','Scratches']
        cfg_path = '/home/huemorgen/Defect-Detection-with-Mixture-of-Experts/configs/swin/swin_tiny.py'
        ckpt = '/home/huemorgen/Defect-Detection-with-Mixture-of-Experts//weights/epoch_12.pth'
        self.inferencer = Inferencer(cfg_path, ckpt)
        self._init_defaults()
        cfg = Config.fromfile(cfg_path)
        self._init_cfg(cfg)
        self._init_connections()
        self._init_corruptions()
        

    def _init_defaults(self):
        W = 200
        self.img_size = (W, W)
        self.input_label.setGeometry(QtCore.QRect(50, 110, *self.img_size))
        self.model_label_size  = self.img_size 

        self.model_label.setScaledContents(True)

        self.corrupts = ['none', 'none', 'none']
        self.severity = 1
        self.slider_1.setTickInterval(1)
        self.slider_1.setRange(1,5)
        self.slider_2.setTickInterval(1)
        self.slider_2.setRange(1,5)
        self.slider_3.setTickInterval(1)
        self.slider_3.setRange(1,5)
        self.model_label.setPixmap(QtGui.QPixmap("GUI/images/dl_model.jpg"))
        self.model_label.setScaledContents(True)
        
    def _init_cfg(self, cfg):
        self.cfg = cfg
        dataloader_cfg = cfg.get('val_dataloader')
        self.dataset = DATASETS.build(dataloader_cfg.dataset)
        self.img_id = 0
        self.load_img(self.img_id)
        self.pipeline = dataloader_cfg.dataset.pipeline
        

    def _init_connections(self):
        # self.folder.clicked.connect(self._button_clk)
        self.comboBox_1.currentTextChanged.connect(self.combo_sig1)
        self.comboBox_2.currentTextChanged.connect(self.combo_sig2)
        self.comboBox_3.currentTextChanged.connect(self.combo_sig3)
        self.next.clicked.connect(self.next_img)
        self.back.clicked.connect(self.back_img)
        self.save.clicked.connect(self.save_cfg)
        self.slider_1.valueChanged.connect(self.update_severity)
        self.slider_2.valueChanged.connect(self.update_severity)
        self.slider_3.valueChanged.connect(self.update_severity)
        # self.predict.clicked.connect(self.inferPressed)

    def _init_corruptions(self):
        self._add_curroption_to_comboBox(self.comboBox_1)
        self._add_curroption_to_comboBox(self.comboBox_2)
        self._add_curroption_to_comboBox(self.comboBox_3)

    def update_severity(self, val):
        self.severity = val
        self.apply_corrupts()

    def load_img(self, i):
        self.img_id = i
        data = self.dataset.__getitem__(self.img_id)
        img = data['inputs'].permute(1,2,0).cpu().numpy()
        self.img_with_gt = self._add_datasample(data)
        self.img_ori = img
        self.img_aug = self.img_ori.copy()
        self.apply_corrupts()

    def save_cfg(self):
        # cfg = copy.deepcopy(self.cfg)
        cfg = self.cfg.copy()
        for corrupt in self.corrupts:
            if corrupt=='none' or self.severity==0: continue
            if 'rotation' in corrupt:
                corrupt_dict = dict(
                    type='RotateCenter',
                    max_rotate_degree = 15*self.severity,
                    scaling_ratio_range=(1, 1),
                    max_aspect_ratio = 100,
                    max_translate_ratio = 0.5,
                    max_shear_degree = 0,
                    skip_filter = False,
                    )
            else:
                corrupt_dict = dict(type='Corrupt',
                                    corruption=corrupt,
                                    severity=self.severity)
            cfg.train_dataloader.dataset.pipeline.insert(2,corrupt_dict)
        
        filename = [cor for cor in self.corrupts if cor!='none']
        filename.append(str(self.severity))
        folder_name = '_'.join(filename)
        filename = f'{folder_name}.py'
        parent = osp.dirname(__file__)
        cfgs_name = 'cfgs'
        cfgs_pth = osp.join(parent, cfgs_name)
        folder_pth = osp.join(cfgs_pth, folder_name)
        file_pth = osp.join(folder_pth, filename)
        mmengine.mkdir_or_exist(cfgs_pth)
        mmengine.mkdir_or_exist(folder_pth)
        text = cfg.pretty_text
        with open(file_pth, 'w') as f:
            f.write(text)
        
        img_pth = osp.join(folder_pth, f'{filename.split(".py")[0]}.jpg')
        imwrite(self.img_aug, img_pth)

    def combo_sig1(self, text):
        self.corrupts[0] = text
        self.apply_corrupts()
    
    def combo_sig2(self, text):
        self.corrupts[1] = text
        self.apply_corrupts()
    
    def combo_sig3(self, text):
        self.corrupts[2] = text
        self.apply_corrupts()

    def next_img(self):
        if self.img_id >= (len(self.dataset)-1): return
        self.load_img(self.img_id+1)

    def back_img(self):
        if self.img_id == 0:
            return
        self.load_img(self.img_id-1)
    
    def _add_datasample(self, data):
        img = data['inputs']
        img_np = img.permute(1,2,0).cpu().numpy().astype(np.uint8)

        viz = DetLocalVisualizer()

        instances = data['data_samples'].gt_instances
        instances.bboxes = instances.bboxes.numpy()
        instances.labels = instances.labels.numpy()
        self.classes = self.dataset.METAINFO['classes']
        palette = self.dataset.METAINFO['palette']
        img_with_gt = viz._draw_instances(img_np.copy(),instances, classes=self.classes, palette=palette)
        return img_with_gt

    def inferPressed(self):
        img = self.img_shown.copy()
        img = self.inferencer.infer(img)
        # self.cur_img = img
        # self.cur_label = self.output_label
        # self.cur_label_size = self.output_label_size
        self.update_pixmap(img, label=self.output_label)

    def _add_curroption_to_comboBox(self, comboBox):
        for corruption in corruptions:
            comboBox.addItem(corruption)
    
    def apply_corrupts(self):
        if self.img_ori is None: return
        img = self.img_ori 
        for corruption in self.corrupts:
            if corruption == 'none':
                continue
            img = self.generic_corrupt(img, self.severity, corruption)
        self.update_pixmap(img, True)
    
    def generic_corrupt(self, img, severity, corruption):
        if 'rotation' in corruption:
            img_dict = dict(img=img)
            aug_img_dict = RotateCenter(
                                withbbox=False,
                                rotation_degree = 15 * severity,
                                scaling_ratio_range=(1, 1),
                                max_translate_ratio = 0.5,
                                max_shear_degree = 0,
                                )(img_dict)
            return aug_img_dict['img']
        elif 'motion' in corruption:
            return np.uint8(motion_blur(img, severity))
        else:
            return corrupt(img, severity, corruption)
    def _update_pixmap(self, label, img):
        label.clear()
        qpix = QPixmap.fromImage(QImage(img.data, img.shape[0], img.shape[1], QImage.Format.Format_RGB888))
        label.setPixmap(qpix)
    def update_pixmap(self, img, augmented=False, label=None ):
        # if label is None:
        #     label = self.input_label
        if img.shape[0] != self.img_size[0]:
            img = mmcv.imresize(img, self.img_size)
        self._update_pixmap(self.input_label, img)
        self.img_shown = img
        if not augmented:
            self.img_ori = img
        else:
            self.img_aug = img
        img = self.img_shown.copy()
        img = self.inferencer.infer(img)
        self._update_pixmap(self.output_label, img)
        
            

        
    # def _button_clk(self):
    #     img = self.img_with_gt.copy()
    #     img =  np.uint8(img)
    #     self.update_pixmap(img, augmented=False)

if __name__ == "__main__":
    register_all_modules()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = AugmentGUI()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
