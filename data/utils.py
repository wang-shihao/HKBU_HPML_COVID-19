import random
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Resampler(object):
    def __init__(self):
        pass

    @classmethod
    def resample(self, slices, threshold):
        '''
        Args:
            slices: the list of slices that requires upsampling.
            threshold: the expected number of slices
        '''
        if threshold >= len(slices):
            return self.upsample(slices, threshold)
        else:
            return self.undersample(slices, threshold)

    @staticmethod
    def upsample(slices, threshold=64):
        original_num = len(slices)
        d = threshold - original_num
        tmp = []
        idxs = []
        for _ in range(d):
            idx = random.randint(0, original_num-1)
            idxs.append(idx)
        for idx, value in enumerate(slices):
            tmp.append(value)
            while idx in idxs:
                idxs.remove(idx)
                tmp.append(value)
        return tmp

    @staticmethod
    def undersample(slices, threshold=64):
        original_num = len(slices)
        d = original_num - threshold
        tmp = slices.copy()
        for _ in range(d):
            idx = random.randint(0, len(tmp)-1)
            tmp.pop(idx)
        return tmp