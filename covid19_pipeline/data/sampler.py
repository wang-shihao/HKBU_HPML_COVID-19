import torch
import torchvision
from torchline.data.sampler import SAMPLER_REGISTRY


__all__ = [
    'WeightedRandomSampler',
]


@SAMPLER_REGISTRY.register()
def WeightedRandomSampler(cfg):
    sampler_cfg = cfg.dataloader.sampler

    weights = sampler_cfg.weights
    num_samples = sampler_cfg.num_samples
    replacement = sampler_cfg.replacement
    return torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
