import numpy as np
import os 
import nibabel as nib 
import imageio 
from multiprocessing import Pool


def scandir(nii_path, save_path):
    p = Pool(40)
    for subdir in os.listdir(nii_path):
        if not subdir.startswith('.'):
            nii_dir = os.path.join(nii_path, subdir) 
            filenames = os.listdir(nii_dir) 
 
            for f in filenames:
                p.apply_async(nii2png, (f, subdir, nii_dir, save_path,)) 
                # nii2png(f, subdir, nii_dir, save_path)

    p.close()
    p.join()

                
def nii2png(f, subdir, nii_dir, save_path):
    img_path = os.path.join(nii_dir, f)
    print("Converting {}".format(img_path))
    img = nib.load(img_path)                
    img_fdata = img.get_fdata()
    fname = f.replace('.nii.gz','')            
    cls_path = os.path.join(save_path, subdir)
    png_path = os.path.join(cls_path, fname.split('_')[1], fname.split('_')[1])
    
    if not os.path.exists(png_path):
        os.makedirs(png_path)                
    
    (x,y,z) = img.shape
    for i in range(z):                      
        silce = img_fdata[i, :, :]          
        imageio.imwrite(os.path.join(png_path,'{}.png'.format(str(i).zfill(4))), silce)

 
if __name__ == '__main__':

    nii_path = '/home/datasets/MosMedData/COVID19_1110/studies/'
    save_path = '/home/datasets/MosMedData/COVID19_1110/pngs/'

    scandir(nii_path, save_path)
