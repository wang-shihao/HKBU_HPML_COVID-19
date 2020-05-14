from torchline.data.transforms import TRANSFORMS_REGISTRY
import torchio
import torchio.transforms as iotf
import numpy as np

__all__ = [
    'CTTransforms',
    '_CTTransforms'
]

@TRANSFORMS_REGISTRY.register()
def CTTransforms(cfg):
    is_train = cfg.dataset.is_train
    slice_num = cfg.dataset.slice_num
    img_size = cfg.input.size
    randomflip = cfg.transforms.ct.randomflip
    randomaffine = cfg.transforms.ct.randomaffine
    randomblur = cfg.transforms.ct.randomblur
    randomnoise = cfg.transforms.ct.randomnoise
    randomswap = cfg.transforms.ct.randomswap
    randomelasticdeformation = cfg.transforms.ct.randomelasticdeformation
    return _CTTransforms(
        is_train = is_train,
        slice_num = slice_num,
        img_size = img_size,
        randomflip = randomflip,
        randomaffine = randomaffine,
        randomblur = randomblur,
        randomnoise = randomnoise,
        randomswap = randomswap,
        randomelasticdeformation = randomelasticdeformation
    )
    

class _CTTransforms(object):
    '''built on torchio
        https://torchio.readthedocs.io/index.html
    '''
    def __init__(self, is_train, slice_num, img_size,
                 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5],
                 randomflip={'enable':0, 'axes':(1,2), 'p':0.5, 'flip_probability':0.5},
                 randomaffine={'enable':0, 'scales':(0.5,0.5), 'degress': (-10,10), 'isotropic': True, 'p': 0.5},
                 randomblur={'enable':0, 'std':(0,4), 'p':0.5},
                 randomnoise={'enable':0, 'mean':(0,0.25), 'std':(0,0.25), 'p':0.5},
                 randomswap={'enable':0, 'patch_size': (16,16,16), 'num_iterations':100, 'p':0.5},
                 randomelasticdeformation={'enable':0, 'num_control_points':(4,4,4),
                                        'max_displacement':(7.5,7.5,7.5), 'locked_borders':2, 'p':0.5},
                 *args, **kwargs):
        self.is_train = is_train
        self.slice_num = slice_num
        self.img_size = img_size
        self.randomflip = randomflip
        self.randomaffine = randomaffine
        self.randomblur = randomblur
        self.randomnoise = randomnoise
        self.randomswap = randomswap
        self.randomelasticdeformation = randomelasticdeformation
        if isinstance(self.img_size, list) or isinstance(self.img_size, tuple):
            self.volume_size = (slice_num, *img_size)
        elif isinstance(self.img_size, int):
            self.volume_size = (slice_num, img_size, img_size)
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
        transform = iotf.Compose([
            iotf.CropOrPad(self.volume_size, padding_mode='edge'),
            #iotf.ZNormalization()
        ])
        return transform

    @property
    def train_transform(self):
        tf_list = [iotf.CropOrPad(self.volume_size, padding_mode='edge')]
        if self.randomflip['enable']:
            params = {key:val for key,val in self.randomflip.items() if key != 'enable'}
            tf_list.append(iotf.RandomFlip(**params))
        if self.randomaffine['enable']:
            params = {key:val for key,val in self.randomaffine.items() if key != 'enable'}
            tf_list.append(iotf.RandomAffine(**params))
        if self.randomblur['enable']:
            params = {key:val for key,val in self.randomblur.items() if key != 'enable'}
            tf_list.append(iotf.RandomBlur(**params))
        if self.randomnoise['enable']:
            params = {key:val for key,val in self.randomnoise.items() if key != 'enable'}
            tf_list.append(iotf.RandomNoise(**params))
        if self.randomswap['enable']:
            params = {key:val for key,val in self.randomswap.items() if key != 'enable'}
            tf_list.append(iotf.RandomSwap(**params))
        if self.randomelasticdeformation['enable']:
            params = {key:val for key,val in self.randomelasticdeformation.items() if key != 'enable'}
            tf_list.append(iotf.RandomElasticDeformation(**params))
        #tf_list.append(iotf.ZNormalization())
        transform = iotf.Compose(tf_list)
        return transform
