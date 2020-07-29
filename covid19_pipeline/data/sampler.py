import torch
import torchvision
from torchline.data.sampler import SAMPLER_REGISTRY
from torchline.data import build_data


__all__ = [
    'WeightedRandomSampler',
]


@SAMPLER_REGISTRY.register()
def WeightedRandomSampler(cfg):
    dataset = build_data(cfg)
    sampler_cfg = cfg.dataloader.sampler

    weights = []
    weights_cls = cfg.dataloader.sampler.weights_cls
    num_samples = len(dataset)
    for i in range(num_samples):
        weight = weights_cls[int(dataset.samples[i]['label'])]
        weights.append(weight)
    replacement = sampler_cfg.replacement
    return torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)