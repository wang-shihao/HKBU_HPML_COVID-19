import json
import os
import random

import cv2
import torch
import torchvision.transforms as TF
from torchline.data import (DATASET_REGISTRY, build_label_transforms,
                            build_transforms)

from .utils import SystematicResampler, RandomResampler, pil_loader


__all__ = [
    'CTDataset',
    '_CTDataset'
]

@DATASET_REGISTRY.register()
def CTDataset(cfg):
    slice_num = cfg.dataset.slice_num
    root_dir = cfg.dataset.dir
    is_train = cfg.dataset.is_train
    is_color = cfg.dataset.is_color
    is_3d = cfg.dataset.is_3d
    if is_train:
        data_list = cfg.dataset.train_list
    else:
        data_list = cfg.dataset.test_list
    if 'Albumentation' in cfg.transforms.name:
        loader = cv2.imread
    else:
        loader = pil_loader
    img_size = cfg.input.size
    transforms = build_transforms(cfg)
    label_transforms = build_label_transforms(cfg)
    return _CTDataset(root_dir, data_list, is_train, is_color, is_3d, img_size, slice_num, loader,
                    transforms, label_transforms)

class _CTDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_list, is_train, is_color=True, is_3d=True, img_size=[224,224], slice_num=64, loader=pil_loader,
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
        self.is_color = is_color
        self.is_3d = is_3d
        self.img_size = img_size
        self.slice_num = slice_num
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.loader = loader
        self.args = args
        self.kwargs = kwargs
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
                        if len(slices)>0:
                            samples[idx] = {'slices':slices, 'label': label, 'path': scan_path}
                            idx += 1
        return samples

    def preprocessing(self, img):
        resize = int(self.img_size[0]*5/4)
        transform = TF.Compose([
            TF.Resize((resize, resize)),
            TF.CenterCrop(self.img_size),
            TF.ToTensor()
        ])
        return transform(img)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = torch.tensor(sample['label']).long()
        if self.is_train:
            slices = RandomResampler.resample(sample['slices'], self.slice_num)
        else:
            slices = SystematicResampler.resample(sample['slices'], self.slice_num)
        path = sample['path']
        slice_tensor = []

        # stack slice
        for slice_ in slices:
            slice_path = os.path.join(path, slice_)
            img = self.loader(slice_path) # height * width * 3
            img = self.preprocessing(img)
            if not self.is_color:
                img = torch.unsqueeze(img[0, :, :], dim=0)
            slice_tensor.append(img)
        slice_tensor = torch.stack(slice_tensor)
        slice_tensor = slice_tensor.permute(1, 0, 2, 3) # c*d*h*w

        # transform
        if self.transforms: slice_tensor = self.transforms.transform(slice_tensor)
        slice_tensor = (slice_tensor-slice_tensor.mean())/(slice_tensor.std()+1e-5)
        if self.label_transforms: label = self.label_transforms.transform(label)

        # if not 3d, then remove channel dimension
        if not self.is_3d: slice_tensor = slice_tensor[0, :, :, :]
        return slice_tensor, label, sample['path']

    def __len__(self):
        return len(self.samples)
