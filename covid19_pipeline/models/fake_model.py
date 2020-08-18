import torch
import torch.nn as nn

from torchline.models import META_ARCH_REGISTRY

__all__ = [
    'FakeNet3D',
    '_FakeNet3D',
]

@META_ARCH_REGISTRY.register()
def FakeNet3D(cfg):
    if cfg.dataset.is_3d:
        c_in = cfg.model.n_input_channels
    else:
        c_in = cfg.dataset.slice_num
    return _FakeNet3D(c_in)


class _FakeNet3D(nn.Module):
    def __init__(self, c_in, classes=3):
        super(_FakeNet3D, self).__init__()
        self.conv = nn.Conv3d(c_in, 16, 3)
        self.glob_avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16,3)
    
    def forward(self, x):
        bs = x.shape[0]
        out = self.conv(x)
        out = self.glob_avgpool(out)
        out = self.fc(out.view(bs, -1))
        return out