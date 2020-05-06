from os import listdir
from os.path import isfile, join
import numpy as np

def get_filelist(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    return onlyfiles

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



