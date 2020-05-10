from torchline.data.transforms import TRANSFORMS_REGISTRY
import torchvision as tv
import torchvision.transforms.functional as TF
import numpy as np

__all__ = [
    'CTTransforms',
    '_CTTransforms'
]

@TRANSFORMS_REGISTRY.register()
def CTTransforms(cfg):
    is_train = cfg.dataset.is_train
    img_size = cfg.input.size

    img_tf = cfg.transforms.img
    # rotate
    is_rotate = img_tf.random_rotation.enable
    rotate_degrees = img_tf.random_rotation.degrees
    # flip
    vflip = img_tf.random_vertical_flip.enable
    hflip = img_tf.random_horizontal_flip.enable
    return _CTTransforms(is_train, img_size)

class _CTTransforms(object):
    def __init__(self, is_train, img_size,
                 is_rotate=False, rotate_degrees=0,
                 hflip=False, vflip=False,
                 is_htrans=False, htrans=0,
                 is_vtrans=False, vtrans=0,
                 is_bright=False, brightness_factor=1,
                 is_contrast=False, contrast_factor=1,
                 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5],
                 *args, **kwargs):
        self.is_train = is_train
        self.img_size = img_size
        self.normsize = tv.transforms.Normalize(mean, std)
        self.transform = self.get_transform()

    def get_transform(self):
        if not self.is_train: 
            print('Generating validation transform ...')
            transform = self.valid_transform
            print(f'Valid transform={transform}')
        else:
            print('Generating training transform ...')
            transform = self.train_transform
            print(f'Train transform={transform}')
        return transform

    @property
    def valid_transform(self):
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.img_size),
            tv.transforms.ToTensor(),
            self.normsize
        ])
        return transform

    @property
    def train_transform(self):
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.img_size),
            tv.transforms.ToTensor(),
            self.normsize
        ])
        return transform
        
    def rotate(self, degree=10):
        pass

    def vtranslation(self):
        pass

    def htranslation(self):
        pass

    def hflip(self, is_flip=True):
        pass

    def vflip(self, is_flip=True):
        pass

    def crop(self):
        pass

    def resize(self):
        pass

    def contrast(self):
        pass

    def bright(self):
        pass

    def normsize(self, mean, std):
        pass