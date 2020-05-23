import sys
sys.path.append('..')

import torch

import models
import torchline as tl
from config.config import add_config




cfg = tl.config.get_cfg()
cfg = add_config(cfg)

x = torch.rand(1,16,64,64)

cfg.model.n_input_channels = 16

names = ['resnet2d', 'resnext2d', 'wide_resnet2d']
depths = [18,34,50,101,152]
for name in names:
    try:
        if name != 'resnet2d': depths = [50,101]
        cfg.model.name = name
        for d in depths:
            cfg.model.model_depth = d
            model = tl.models.build_model(cfg)
            y = model(x)
            print(f"{name} {d} pass")
    except Exception as e:
        print(str(e))
        print(f"{name} not pass")
