import numpy as np
import re
import pandas as pd
from skimage import morphology
from skimage import measure
from PIL import Image
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import sys
import os
import cv2
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import multiprocessing



def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)

    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    #print(lungwin)
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')

    return newimg


def savepng(img_to_save, output, name):
    imgs_to_process = Image.open(imgs_to_process)
    imgs_to_process.save(output + name + '.png', 'PNG')


def lungSeg(imgs_to_process,output,name):

    if os.path.exists(output+'/'+name+'_clean.npy') : return
    imgs_to_process = Image.open(imgs_to_process)
    imgs_to_process,_,_ = imgs_to_process.split()
    #imgs_to_process = imgs_to_process

    #imgs_to_process = cv2.imread(imgs_to_process)
    img_to_save = imgs_to_process.copy()
    img_to_save = np.asarray(img_to_save).astype('uint8')

    imgs_to_process = lumTrans(imgs_to_process)    
    imgs_to_process = np.expand_dims(imgs_to_process, axis=0)
    #print(imgs_to_process.shape)
    x,y,z = imgs_to_process.shape 
    #if y!=512 : continue
  
    img_array = imgs_to_process.copy()  
    #print(img_to_save)
    A1 = int(y/(512./100))
    A2 = int(y/(512./400))

    A3 = int(y/(512./475))
    A4 = int(y/(512./40))
    A5 = int(y/(512./470))
    #print "on image", img_file
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        print(img.shape)
        x,y = img.shape
        #Standardize the pixel values
        allmean = np.mean(img)
        allstd = np.std(img)
        img = img-allmean
        img = img/allstd
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[A1:A2,A1:A2] 
        mean = np.mean(middle)  
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        #img[img==max]=mean
        #img[img==min]=mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid 
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region 
        # engulf the vessels and incursions into the lung cavity by 
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions 
        #  with a bbox that is to large in either dimnsion
        #  Also, the lungs are generally far away from the top 
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity. 
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<A3 and B[3]-B[1]<A3 and B[0]>A4 and B[2]<A5:
                good_labels.append(prop.label)
        mask = np.ndarray([x,y],dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        imgs_to_process[i] = mask

    m1 = imgs_to_process
    
    convex_mask = m1
    dm1 = process_mask(m1)
    dilatedMask = dm1
    Mask = m1
    extramask = dilatedMask ^ Mask
    bone_thresh = 180
    pad_value = 0

    img_array[np.isnan(img_array)]=-2000
    sliceim = img_array
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value

    #sliceim = img_array*imgs_to_process

    x,y,z = sliceim.shape
    #print(output)
    if not os.path.exists(output): 
        os.makedirs(output)
    #print(sliceim.shape)
    #im = Image.fromarray(sliceim.reshape(1, x, y, z))
    #print(sliceim.shape, img_to_save.shape)

    img_to_save[sliceim.squeeze()==0] = 0
    #print(img_to_save)

    #im = Image.fromarray(sliceim.squeeze())
    #im = Image.fromarray(img_to_save.squeeze())
    im = Image.fromarray(img_to_save)

    print(output + name + '.png')

    im.save(output + name + '.png', 'PNG')
    #im.save(os.path.join(output,name+'.png'), 'PNG')
    #np.save(os.path.join(output,name+'_clean.npy'),sliceim.reshape(1,x,y,z))
    #np.save(os.path.join(output,name+'_label.npy'),np.array([[0,0,0,0]]))
    #lb = Image.fromarray(sliceim.reshape(1, x, y, z))
    #lb.save(os.path.join(output,name+'_label.npy'))


if __name__ == "__main__":
    tmpfile = './datalist.txt'
    luna_data = '/home/datasets/CCCCI_cleaned/raw'
    if not os.path.exists(tmpfile):
        splits = os.listdir(luna_data)
        patients = []
        scans = []
        imgs = []
        for split in splits:
            patients = patients + ['{}/{}/{}'.format(luna_data, split, x) for x in os.listdir(os.path.join(luna_data, split))]
        for patient in patients:
            scans = scans + ["{}/{}".format(patient, x) for x in os.listdir(patient)]
        for scan in scans:
            imgs = imgs + ["{}/{}".format(scan, x) for x in os.listdir(scan)]
        f = open(tmpfile, 'w')
        f.writelines(["{}\n".format(img) for img in imgs])
        f.close()

    else:
        f = open(tmpfile, 'r')
        imgs = [a.strip() for a in f.readlines()]

    filelist = imgs

    p = multiprocessing.Pool(20)
    #ref = pd.read_csv('CT_cleaned_v2.csv').fillna(0)
    ref = pd.read_csv('new_missed.csv').fillna(0)
    for _, line in ref.iterrows():
        #print(line)
        pid = line['pid']
        cls = line.cls
        scan_id = line.scan_id
        is_seg = line.is_seg
        rawdir = "/home/datasets/CCCCI_cleaned/raw/{}/{}/{}".format(cls, pid, scan_id)
        print(rawdir)
        for rimg in os.listdir(rawdir):
            print(rimg)
            img_path = os.path.join(rawdir, rimg)
            output = rawdir.replace('raw', 'dataset_cleaned') + '/'
            name = rimg.split('.')[0]
            print("Processing {}".format(img_path))
            if is_seg == 1:
                #p.apply_async(savepng, (img_path, output, name,))
                savepng(img_path, output, name)
            else:
                #p.apply_async(lungSeg, (img_path, output, name,))
                lungSeg(img_path, output, name)

    p.close()
    p.join()



