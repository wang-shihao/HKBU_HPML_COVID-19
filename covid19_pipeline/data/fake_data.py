import torch
from torchline.data import DATASET_REGISTRY

__all__ = [
    'FakeData',
    '_FakeData'
]

class _FakeData(torch.utils.data.Dataset):
    def __init__(self, channels=1, size=[64,64], num=4):
        if isinstance(size, int):
            self.size = [size, size]
        elif isinstance(size, list):
            self.size = size
        self.num = num
        self.data = torch.rand(num, channels, 16, *size)
        self.labels = torch.randint(0, 3, (num,))


    def __getitem__(self, index):
        return self.data[index], self.labels[index], ''

    def __len__(self):
        return self.num

@DATASET_REGISTRY.register()
def FakeData(cfg):
    size = cfg.input.size
    channels = cfg.model.n_input_channels
    return _FakeData(channels, size)
