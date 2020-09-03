from scipy import misc
import numpy as np

path='/home/comp/csshshi/tmp'
idx = 0 
crop_size = 96

img = np.load('%s/img_%d.npy'% (path, idx))
coord = np.load('%s/coord_%d.npy'% (path, idx))
label = np.load('%s/label_%d.npy'% (path, idx))
print 'coord shape: ', coord.shape

def get_original(pos):
    return int((pos+0.5) * crop_size)

z=coord[0][0][0][0]
x=coord[1][0][0][0]
y=coord[2][0][0][0]
print x, y, z
x = get_original(x)
y = get_original(y)
z = get_original(z)
print x, y, z

img = img*128+128
img =img[0][z]
print img.shape

misc.imsave('img.jpg',img)

#chunk_img =img[(x-20):(x+20),(y-20):(y+20)]
#misc.imsave('chunk.jpg',chunk_img)


