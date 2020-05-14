import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet

from torchline.models import META_ARCH_REGISTRY


__all__ = ['wide_resnet3d']

class WideBottleneck(resnet.Bottleneck):
    expansion = 2


def generate_model(model_depth, k, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    inplanes = [x * k for x in resnet.get_inplanes()]
    if model_depth == 50:
        model = resnet.ResNet(WideBottleneck, [3, 4, 6, 3], inplanes, **kwargs)
    elif model_depth == 101:
        model = resnet.ResNet(WideBottleneck, [3, 4, 23, 3], inplanes, **kwargs)
    elif model_depth == 152:
        model = resnet.ResNet(WideBottleneck, [3, 8, 36, 3], inplanes, **kwargs)
    elif model_depth == 200:
        model = resnet.ResNet(WideBottleneck, [3, 24, 36, 3], inplanes,
                              **kwargs)

    return model

@META_ARCH_REGISTRY.register()
def wide_resnet3d(cfg):
    return generate_model(cfg)
