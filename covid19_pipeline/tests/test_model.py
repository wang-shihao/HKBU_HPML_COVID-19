import sys
sys.path.append('..')
import torchline as tl
from config.config import add_config
import models

cfg = tl.config.get_cfg()
cfg = add_config(cfg)
cfg.merge_from_file('../config/config.yml')

model_names = {
    'mc3_18': [],
    'r3d_18': [],
    'r2plus1d_18': [],
    'densenet3d': [121,169,201,264],
    'resnet3d': [10,18,34,50,101,152,200],
    # 'wide_resnet3d': [50,101,152,200],
    'resnext3d': [50,101,152,200],
    'preact_resnet3d': [10,18,34,50,101,152,200],
}

for name in model_names:
    depths = model_names[name]
    if name == 'densenet3d':
        cfg.model.model_depth = 121
    if name in ['resnet3d',
        'wide_resnet3d',
        'resnext3d',
        'preact_resnet3d']:
        cfg.model.model_depth = 101
    try:
        cfg.model.name = name
        if len(depths)>=1:
            for depth in depths:
                cfg.model.model_depth = depth
                model = tl.models.build_model(cfg)
                model_size = tl.utils.model_size(model)
                print(f"{name}, depth={depth} , size={model_size}MB")
        else:
            model = tl.models.build_model(cfg)
            model_size = tl.utils.model_size(model)
            print(f"{name}, size={model_size}MB")
    except Exception as e:
        # cfg.model.name = name
        # model = tl.models.build_model(cfg)
        # print(name, tl.utils.model_size(model),'MB')
        print(name, 'not pass')