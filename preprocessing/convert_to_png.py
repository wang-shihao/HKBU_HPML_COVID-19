import os

from PIL import Image


def savepng(imgs_to_save, output, name):
    imgs_to_save = Image.open(imgs_to_save)
    if not os.path.exists(output):
        os.makedirs(output)
    imgs_to_save.save(output + name + '.png', 'PNG')

def main():
    src_path = '/home/datasets/CCCCI_cleaned/raw'
    dst_path = '/home/datasets/CCCCI_cleaned/dataset_cleaned'

    for c in os.listdir(src_path): # ['CP',"NCP", "Normal"]
        p_path = os.path.join(src_path, c)
        for pid in os.listdir(p_path):
            scan_path = os.path.join(p_path, pid)
            for scan_id in os.listdir(scan_path):
                pass


if __name__ == '__main__':
    main()

