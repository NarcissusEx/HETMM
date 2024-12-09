import os
from PIL import Image
import numpy as np
from functools import reduce
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as ts

class Base(Dataset):

    def __init__(self, params):
        for key, value in params.items():
            self.__setattr__(key, value)
        mean = self.normalize['mean']
        std = self.normalize['std']
        self.img_trans = ts.Compose(
            [
                ts.Resize(self.out_size),
                ts.ToTensor(),
                ts.Normalize(**self.normalize)
            ]
        )
        self.gt_trans = ts.Compose(
            [
                ts.Resize(self.out_size),
                ts.ToTensor()
            ]
        )
        self.inv_trans = ts.Compose(
            [
                ts.Normalize(
                    mean=[-m / s for m, s in zip(mean, std)], 
                    std=[1. / s for s in std]
                )
            ]
        )
        self.load_data()

    def __len__(self):
        return len(self.img_paths)
    
    def _get_gt(self, idx):
        gt = self.gt_paths[idx]
        return torch.zeros([1, *self.out_size]) if gt == 0 else self.gt_trans(Image.open(gt).convert('L'))
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.img_trans(Image.open(img_path).convert('RGB'))
        gt = self._get_gt(idx)
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt
    
class MVTec_AD(Base):

    def __init__(self, **params):
        super(MVTec_AD, self).__init__(params)

    def load_data(self):
        self.img_path = os.path.join(self.root, self.category, self.mode if self.mode == 'test' else 'train')
        if self.mode == 'test':
            self.gt_path = os.path.join(self.root, self.category, 'ground_truth')
        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []
        for defect_type in filter(lambda x : os.path.isdir(os.path.join(self.img_path, x)), os.listdir(self.img_path)):
            img_paths = [os.path.join(self.img_path, defect_type, x) for x in sorted(filter(lambda x : x.endswith('.png'), \
                os.listdir(os.path.join(self.img_path, defect_type))))]
            gt_paths = [os.path.join(self.gt_path, defect_type, x) for x in sorted(filter(lambda x : x.endswith('.png'), \
                os.listdir(os.path.join(self.gt_path, defect_type))))] if defect_type != 'good' else [0] * len(img_paths)
            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
            self.labels.extend([int(defect_type != 'good')] * len(img_paths))
            self.types.extend([defect_type] * len(img_paths))
        self.filenames = [os.path.basename(reduce(lambda x1, x2 : x1 + '.' + x2, img_path.split('.')[:-1])) for img_path in self.img_paths]

        assert len(self.img_paths) == len(self.gt_paths), "Something wrong with test and ground truth pair!"