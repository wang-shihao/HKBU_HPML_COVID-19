
import torch
import torch.nn as nn
from torchvision import models

from torchline.models import META_ARCH_REGISTRY

__all__ = [
    'mc3_18',
    'r3d_18',
    'r2plus1d_18'
]

class VideoResNet(nn.Module):

    def __init__(self, model):
        """Generic resnet video generator.
        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        for key, val in model.__dict__.items():
            self.__dict__[key] = val
        self.g_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.last_linear = self.fc

    def features(self, x):
        out = self.stem(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def logits(self, x):
        out = self.g_avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.last_linear(out)
        return out
        
    def forward(self, x):
        out = self.features(x)
        out = self.logits(out)
        return out

def generate_model(cfg, name):
    pretrained=cfg.model.pretrained
    classes = cfg.model.classes
    model = eval(f"models.video.{name}(pretrained={pretrained})")
    if classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, classes, bias=False)
    return VideoResNet(model)

@META_ARCH_REGISTRY.register()
def mc3_18(cfg):
    return generate_model(cfg, 'mc3_18')

@META_ARCH_REGISTRY.register()
def r3d_18(cfg):
    return generate_model(cfg, 'r3d_18')

@META_ARCH_REGISTRY.register()
def r2plus1d_18(cfg):
    return generate_model(cfg, 'r2plus1d_18')