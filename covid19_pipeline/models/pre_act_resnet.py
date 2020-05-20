import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import conv3x3x3, conv1x1x1, get_inplanes, ResNet

from torchline.models import META_ARCH_REGISTRY


__all__ = ['preact_resnet3d']


class PreActivationBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreActivationBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(PreActivationBasicBlock, [1, 1, 1, 1], get_inplanes(),
                       **kwargs)
    elif model_depth == 18:
        model = ResNet(PreActivationBasicBlock, [2, 2, 2, 2], get_inplanes(),
                       **kwargs)
    elif model_depth == 34:
        model = ResNet(PreActivationBasicBlock, [3, 4, 6, 3], get_inplanes(),
                       **kwargs)
    elif model_depth == 50:
        model = ResNet(PreActivationBottleneck, [3, 4, 6, 3], get_inplanes(),
                       **kwargs)
    elif model_depth == 101:
        model = ResNet(PreActivationBottleneck, [3, 4, 23, 3], get_inplanes(),
                       **kwargs)
    elif model_depth == 152:
        model = ResNet(PreActivationBottleneck, [3, 8, 36, 3], get_inplanes(),
                       **kwargs)
    elif model_depth == 200:
        model = ResNet(PreActivationBottleneck, [3, 24, 36, 3], get_inplanes(),
                       **kwargs)

    return model

@META_ARCH_REGISTRY.register()
def preact_resnet3d(cfg):
    return generate_model(cfg.model.model_depth, classes=cfg.model.classes, n_input_channels=cfg.model.n_input_channels)
