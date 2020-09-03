import glob
import io
import os

import numpy as np
import PIL.Image as pil_image


def convert(src, des):
    with open(src, 'rb') as f:
        tif = pil_image.open(io.BytesIO(f.read()))
    array=np.array(tif)
    max_val=np.amax(array)
    normalized=(array/max_val)
    im = pil_image.fromarray(normalized)
    im.save(des)

if __name__ == '__main__':
    files = glob.glob('./COVID-CTset/*/*/*/*.tif')
    for file in files:
        src = file
        des = file.replace('COVID-CTset', 'COVID-CTset_visual')
        des_path = os.path.dirname(file).replace('COVID-CTset', 'COVID-CTset_visual')
        if not os.path.exists(des_path): os.makedirs(des_path)
        convert(src, des)