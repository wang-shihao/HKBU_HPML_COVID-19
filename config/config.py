from torchline.config import CfgNode as CN

__all__ = [
    'add_config'
]

def add_config(cfg):
    '''
    cfg.new_item = CN()
    '''
    cfg.dataset.slice_num = 64
    
    cfg.transforms.tensor.normalization.mean = [0.5, 0.5, 0.5]
    cfg.transforms.tensor.normalization.std = [0.5, 0.5, 0.5]

    # resize
    cfg.transforms.img.resize =  CN()
    cfg.transforms.img.resize.enable = 1

    # random_crop
    cfg.transforms.img.random_crop = CN()
    cfg.transforms.img.random_crop.enable = 1

    # color_jitter
    cfg.transforms.img.color_jitter = CN()
    cfg.transforms.img.color_jitter.enable = 0
    cfg.transforms.img.color_jitter.brightness = 0.1
    cfg.transforms.img.color_jitter.contrast = 0.1
    cfg.transforms.img.color_jitter.saturation = 0.1
    cfg.transforms.img.color_jitter.hue = 0.1

    # horizontal_flip
    cfg.transforms.img.random_horizontal_flip = CN()
    cfg.transforms.img.random_horizontal_flip.enable = 1
    cfg.transforms.img.random_horizontal_flip.p = 0.5

    # vertical_flip
    cfg.transforms.img.random_vertical_flip = CN()
    cfg.transforms.img.random_vertical_flip.enable = 1
    cfg.transforms.img.random_vertical_flip.p = 0.5

    # random_rotation
    cfg.transforms.img.random_rotation = CN()
    cfg.transforms.img.random_rotation.enable = 1
    cfg.transforms.img.random_rotation.degrees = 10
    return cfg
