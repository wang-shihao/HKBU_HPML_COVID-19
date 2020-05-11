import json
import os
import random

import cv2
import torch
import torchvision as TF
from torchline.data import (DATASET_REGISTRY, build_label_transforms,
                            build_transforms)

from .utils import Resampler, pil_loader


__all__ = [
    'CTDataset',
    '_CTDataset'
]

@DATASET_REGISTRY.register()
def CTDataset(cfg):
    slice_num = cfg.dataset.slice_num
    root_dir = cfg.dataset.dir
    is_train = cfg.dataset.is_train
    if is_train:
        data_list = cfg.dataset.train_list
    else:
        data_list = cfg.dataset.test_list
    if 'Albumentation' in cfg.transforms.name:
        loader = cv2.imread
    else:
        loader = pil_loader
    transforms = build_transforms(cfg)
    label_transforms = build_label_transforms(cfg)
    return _CTDataset(root_dir, data_list, is_train, slice_num, loader, transforms, label_transforms)

class _CTDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_list, is_train, slice_num=64, loader=pil_loader,
                 transforms=None, label_transforms=None, *args, **kwargs):
        '''
        Args:
            root_dir: root dir of dataset, e.g., ~/../../datasets/CCCCI_cleaned/dataset_cleaned/
            data_list: the training of testing data list or json file. e.g., ct_train.json
            is_train: determine to load which type of dataset
            slice_num: the number of slices in a scan
        '''
        self.root_dir = root_dir
        self.data_list = data_list
        self.is_train = is_train
        self.slice_num = slice_num
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.loader = loader
        with open(self.data_list, 'r') as f:
            self.data = json.load(f)
        self.cls_to_label = {key:idx for idx, key in enumerate(self.data)} # {'CP': 0, 'NCP': 1, 'Normal': 2}
        self.samples = self.convert_json_to_list(self.data)

    def convert_json_to_list(self, data):
        samples = {} # {0: {'scans': [], 'labels': 0}}
        idx = 0
        for cls_ in data:
            for pid in data[cls_]:
                for scan_id in data[cls_][pid]:
                    slices = data[cls_][pid][scan_id]
                    label = self.cls_to_label[cls_]
                    scan_path = os.path.join(self.root_dir,cls_,pid,scan_id)
                    if os.path.exists(scan_path):
                        # for slice_ in slices:
                        #     slice_path = os.path.join(scan_path, slice_)
                        #     if not os.path.exists(slice_path):
                        #         slices.remove(slice_)
                        if len(slices)>0:
                            samples[idx] = {'slices':slices, 'label': label, 'path': scan_path}
                            idx += 1
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = torch.tensor(sample['label']).long()
        slices = Resampler.resample(sample['slices'], self.slice_num)
        path = sample['path']
        slice_tensor = []

        for slice_ in slices:
            slice_path = os.path.join(path, slice_)
            img = self.loader(slice_path) # height * width * 3
            if self.transforms: img = self.transforms.transform(img)
            slice_tensor.append(img)
        slice_tensor = torch.stack(slice_tensor)
        slice_tensor = slice_tensor.permute(1, 0, 2, 3)
        if self.label_transforms: label = self.label_transforms(label)
        return slice_tensor, label

    def __len__(self):
        return len(self.samples)
