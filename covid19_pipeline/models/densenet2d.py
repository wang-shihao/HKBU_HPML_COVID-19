import torch
import torch.nn as nn
import torchvision as tv

from torchline.models import META_ARCH_REGISTRY

def modify_in_channels(model, n_input_channels):
    conv0 = model.features.conv0
    in_channels = conv0.in_channels
    out_channels = conv0.out_channels
    kernel_size = conv0.kernel_size
    model.features.conv0.in_channels = n_input_channels
    model.features.conv0.weight.data = torch.rand(out_channels, n_input_channels, *kernel_size)
    return model

def generate_densenet(model_depth, num_classes, n_input_channels, pretrained=False, **kwargs):
    assert model_depth in [121, 161, 201]
    model = eval(f"tv.models.densenet{model_depth}(pretrained={pretrained}, num_classes={num_classes})")
    if n_input_channels != 3: model = modify_in_channels(model, n_input_channels)
    return model

@META_ARCH_REGISTRY.register()
def densenet2d(cfg):
    model_depth = cfg.model.model_depth
    num_classes = cfg.model.classes
    n_input_channels = cfg.model.n_input_channels
    return generate_densenet(model_depth=model_depth,
                     num_classes=num_classes,
                     n_input_channels=n_input_channels)