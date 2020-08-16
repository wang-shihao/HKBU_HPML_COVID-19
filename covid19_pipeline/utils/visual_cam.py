import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize, rotate
from torchvision import transforms as TF


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CAM:
    def __init__(self, cfg, model):
        '''
        model: 1) glob_avgpool 2) fc
        '''
        self.cfg = cfg
        self.model = model
        self.parse_cfg()

    def parse_cfg(self):
        self.scan_path = self.cfg.cam.scan_path
        self.label = self.cfg.cam.label
        self.pool_name = self.cfg.cam.pool_name
        self.save_path = self.cfg.cam.save_path
        self.is_color = self.cfg.dataset.is_color
        self.img_size = self.cfg.input.size[0]
        self.loader = pil_loader
        resize = int(self.img_size*5/4)
        self.transform = TF.Compose([
            TF.Resize((resize, resize)),
            TF.CenterCrop(self.img_size),
            TF.ToTensor()
        ])

    def run(self):
        # data preprocessing
        scan = self.data_preprocess(self.scan_path, self.transform, self.is_color, 
                                    self.img_size, self.loader)
        bs, channel, depth, height, width = scan.shape

        # model
        model = self.register_hook(self.model, self.pool_name)
        preds = model(scan)

        # post-process
        params = list(model.parameters()) # params of last linear layer
        weight = np.squeeze(params[-2].data.cpu().detach().numpy()) # class*c
        feat_maps = self.activation['map'][0].cpu().detach().numpy() # 1*c*d*h*w
        feat_num = weight.shape[1]
        feat_size = feat_maps.shape[2:]
        idx = self.label if self.label else preds.argmax(1)
        weight_map = np.dot(weight[idx].reshape(1, feat_num),
                            feat_maps[0].reshape(feat_num, -1)).reshape(1, *feat_size) # d*h*w
        weight_map = (weight_map - np.min(weight_map)) / np.max(weight_map)
        weight_map = resize(weight_map[0], (depth, width, height))
        result = weight_map * scan.numpy()[0,0,::] # depth, height, width
        self.visualize(scan.numpy()[0,0,::], weight_map, 2)
        # weight_map = np.uint8(255*weight_map)
        
        # cam = resize(weight_map[0], (depth, width, height))
        # heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # img_name = img_path.replace('.jpg', '_cam.jpg')
        # img_name = os.path.join(self.save_path, img_name)
        # cv2.imwrite(img_name, result)


    def data_preprocess(self, scan_path, transform=None, is_color=False,
                        img_size=128, loader=pil_loader):
        if self.cfg.cam.debug:
            return torch.rand(1,1,64,32,32)
        if not transform: transform = self.transform
        scan = []
        for slice_img in os.listdir(scan_path):
            img = loader(img)
            img = transform(img) # 3*h*w
            if is_color: img = torch.unsqueeze(img[0, :, :], dim=0) # 1*h*w
            scan.append(img)
        scan = torch.stack(scan) # d*c*h*w
        scan = scan.permute(1, 0, 2, 3) # c*d*h*w
        scan = torch.unsqueeze(scan, 0)
        return scan

    def register_hook(self, model, pool_name=None):
        self.activation = {}
        def hook(model, input, output):
            self.activation['map'] = input
        if pool_name:
            eval(f"model.model.{pool_name}.register_forward_hook(hook)")
        else:
            try:
                model.model.glob_avgpool.register_forward_hook(hook)
            except:
                raise NotImplementedError('global avgpool not found.')
        return model

    def visualize(self, scan_img, cam, slice_id=16):
        depth, height, width = scan_img.shape
        fig = plt.figure(figsize=(21, 5))
        plt.subplot(1, 3, 1)
        plt.axis('off')
        matr = rotate(scan_img[:, :, slice_id].reshape(depth, height), 90)
        plt.imshow(matr, cmap=plt.cm.Greys_r, interpolation=None,
                vmax=1., vmin=0.)
        # plt.hold(True)
        matr = rotate(cam[:, :, slice_id].reshape(depth, height), 90)
        plt.imshow(matr,
                interpolation=None, vmax=1., vmin=.0, alpha=1,
                cmap=plt.cm.rainbow)
        plt.subplot(1, 3, 2)
        plt.axis('off')
        matr = rotate(scan_img[slice_id,:, :].reshape(height, width), 90)
        plt.imshow(matr, cmap=plt.cm.Greys_r, interpolation=None,
                vmax=1., vmin=0.)
        # plt.hold(True)
        matr = rotate(cam[slice_id,:, :].reshape(height, width), 90)
        plt.imshow(matr,
                interpolation=None, vmax=1., vmin=.0, alpha=1,
                cmap=plt.cm.rainbow)
        plt.subplot(1, 3, 3)
        plt.axis('off')
        matr = rotate(scan_img[:,slice_id, :].reshape(depth, width), 90)
        plt.imshow(matr, cmap=plt.cm.Greys_r, interpolation=None,
                vmax=1., vmin=0.)
        # plt.hold(True)
        matr = rotate(cam[:,slice_id, :].reshape(depth, width), 90)
        plt.imshow(matr,
                interpolation=None, vmax=1., vmin=.0, alpha=1,
                cmap=plt.cm.rainbow)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        plt.show()

def register_hook(model, pool_name=None):
    activation = {}
    def hook(model, input, output):
        activation['map'] = input
    if pool_name:
        eval(f"model.{pool_name}.register_forward_hook(hook)")
    else:
        try:
            model.avg_pool.register_forward_hook(hook)
        except:
            raise NotImplementedError('global avgpool not found.')
    return model

def gen_heatmap(self, img_paths, labels=None, transform=None,):
    '''
    img_paths: the list of all image paths 
    '''
    if not transform:
        resize = int(self.img_size[0]*5/4)
        transform = TF.Compose([
        TF.Resize((resize, resize)),
        TF.CenterCrop(self.img_size),
        TF.ToTensor()
    ])
    for img_path in img_paths:
        img = pil_loader(img_path)
        img = transform(img)

def visualize(cfg, model, img_paths):
    model = register_hook(model)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.6467, 0.4797, 0.4392], [0.1923, 0.1741, 0.1711])
    ])
    # render the CAM and output
    for img_path in img_paths:
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        timg = torch.tensor(img).permute(2,0,1)
        timg = preprocess(timg).unsqueeze(0).cuda()
        preds = model(timg) # bs * 10
        params = list(model.parameters())
        weight = np.squeeze(params[-2].data.cpu().detach().numpy()) # 1000*2048
        feat_maps = activation['map'][0].cpu().detach().numpy() # 1*2048*14*14
        feat_num = weight.shape[1]
        feat_wh = feat_maps.shape[-1]
        idx = preds.argmax(1)
        weight_map = np.dot(weight[idx].reshape(1, feat_num), feat_maps[0].reshape(feat_num, -1)).reshape(1, feat_wh, feat_wh) # 1*14*14
        weight_map = (weight_map - np.min(weight_map)) / np.max(weight_map)
        weight_map = np.uint8(255*weight_map)
        
        CAM = cv2.resize(weight_map[0], (width, height))
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        img_name = img_path.replace('.jpg', '_cam.jpg')
        cv2.imwrite(img_name, result)

def data_preprocess(scan_path, transform, is_color, img_size, loader):
    if not transform:
        resize = int(img_size*5/4)
        transform = TF.Compose([
            TF.Resize((resize, resize)),
            TF.CenterCrop(img_size),
            TF.ToTensor()
        ])
    scan = []
    for slice_img in os.listdir(scan_path):
        img = loader(img)
        img = transform(img) # 3*h*w
        if is_color: img = torch.unsqueeze(img[0, :, :], dim=0) # 1*h*w
        scan.append(img)
    scan = torch.stack(scan) # d*c*h*w
    scan = scan.permute(1, 0, 2, 3) # c*d*h*w
    return scan

def gen_cam_heatmap(model, scan_path, label=None, transform=None, pool_name=None,
                    is_color=False, loader=pil_loader, img_size=128):
    '''
    model:
        1) model.glob_avgpool
        2) model.fc
    '''
    # data preprocessing
    scan = data_preprocess(scan_path, transform, is_color, img_size, loader)

    # model
    model = register_hook(model, pool_name)
    preds = model(scan)

    # post-process
    params = list(model.parameters()) # params of last linear layer
    weight = np.squeeze(params[-2].data.cpu().detach().numpy()) # 1000*2048
    feat_maps = activation['map'][0].cpu().detach().numpy() # 1*2048*14*14
    feat_num = weight.shape[1]
    feat_wh = feat_maps.shape[-1]
    idx = label if label else preds.argmax(1)
    weight_map = np.dot(weight[idx].reshape(1, feat_num), feat_maps[0].reshape(feat_num, -1)).reshape(1, feat_wh, feat_wh) # 1*14*14
    weight_map = (weight_map - np.min(weight_map)) / np.max(weight_map)
    weight_map = np.uint8(255*weight_map)
    
    CAM = cv2.resize(weight_map[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    img_name = img_path.replace('.jpg', '_cam.jpg')
    cv2.imwrite(img_name, result)
