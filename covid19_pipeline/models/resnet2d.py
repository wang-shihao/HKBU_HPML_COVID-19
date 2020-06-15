import torch
import torch.nn as nn
import torchvision as tv

from torchline.models import META_ARCH_REGISTRY

def modify_in_channels(model, n_input_channels):
    conv1 = model.conv1
    in_channels = conv1.in_channels
    out_channels = conv1.out_channels
    kernel_size = conv1.kernel_size
    model.conv1.in_channels = n_input_channels
    model.conv1.weight.data = torch.rand(out_channels, n_input_channels, *kernel_size)
    return model

def generate_resnet(model_depth, num_classes, n_input_channels, pretrained=False, **kwargs):
    assert model_depth in [18, 34, 50, 101, 152]
    model = eval(f"tv.models.resnet{model_depth}(pretrained={pretrained}, num_classes={num_classes})")
    if n_input_channels != 3: model = modify_in_channels(model, n_input_channels)
    return model

def generate_resnext(model_depth, num_classes, n_input_channels, pretrained=False, **kwargs):
    assert model_depth in [50, 101]
    if model_depth == 50:
        model = tv.models.resnext50_32x4d(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_depth == 101:
        model = tv.models.resnext101_32x8d(pretrained=pretrained, num_classes=num_classes, **kwargs)
    if n_input_channels != 3: model = modify_in_channels(model, n_input_channels)
    return model

def generate_wide_resnet(model_depth, num_classes, n_input_channels, pretrained=False, **kwargs):
    assert model_depth in [50, 101]
    if model_depth == 50:
        model = tv.models.wide_resnet50_2(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_depth == 101:
        model = tv.models.wide_resnet101_2(pretrained=pretrained, num_classes=num_classes, **kwargs)
    if n_input_channels != 3: model = modify_in_channels(model, n_input_channels)
    return model

@META_ARCH_REGISTRY.register()
def resnet2d(cfg):
    model_depth = cfg.model.model_depth
    num_classes = cfg.model.classes
    n_input_channels = cfg.model.n_input_channels
    return generate_resnet(model_depth=model_depth,
                     num_classes=num_classes,
                     n_input_channels=n_input_channels)

@META_ARCH_REGISTRY.register()
def resnext2d(cfg):
    model_depth = cfg.model.model_depth
    num_classes = cfg.model.classes
    n_input_channels = cfg.model.n_input_channels
    return generate_resnet(model_depth=model_depth,
                     num_classes=num_classes,
                     n_input_channels=n_input_channels)

@META_ARCH_REGISTRY.register()
def wide_resnet2d(cfg):
    model_depth = cfg.model.model_depth
    num_classes = cfg.model.classes
    n_input_channels = cfg.model.n_input_channels
    return generate_wide_resnet(model_depth=model_depth,
                     num_classes=num_classes,
                     n_input_channels=n_input_channels)