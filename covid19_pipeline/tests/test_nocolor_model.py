import sys
sys.path.append('..')
import torchline as tl
from config.config import add_config
import models
import torch

cfg = tl.config.get_cfg()
cfg = add_config(cfg)
cfg.merge_from_file('../config/config.yml')
cfg.model.n_input_channels = 1

x = torch.rand(1,1,32,64,64)

model_names = {
    'mc3_18': [],
    # 'r3d_18': [],
    # 'r2plus1d_18': [],
    'densenet3d': [121,169,201,264],
    # 'resnet3d': [10,18,34,50,101,152,200],
    # 'wide_resnet3d': [50,101,152,200],
    # 'resnext3d': [50,101,152,200],
    # 'preact_resnet3d': [10,18,34,50,101,152,200],
}

for name in model_names:
    depths = model_names[name]
    if name == 'densenet3d':
        cfg.model.model_depth = 121
    if name in ['resnet3d',
        'wide_resnet3d',
        'resnext3d',
        'preact_resnet3d']:
        cfg.model.model_depth = 50
    try:
        cfg.model.name = name
        net = tl.models.build_model(cfg)
        y = net(x)
        print(f"{name} pass")
    except Exception as e:
        cfg.model.name = name
        net = tl.models.build_model(cfg)
        y = net(x)
        print(name, tl.utils.model_size(model),'MB')
        print(name, 'not pass')