import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import settings
import utils
import imageio
import os
import plotly.graph_objects as go


def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    ax.set_xlabel('2^M')
    ax.set_ylabel('2^N')
    ax.set_zlabel('2^K')
#ax.scatter(x, y, z, c=plt.cm.gray(cNorm(cs)))
    scalarMap.set_array(cs)
    plt.show()



def plot_size_hist():
    sizelist = np.fromfile('size.npy', type=np.float32)


def show_img(file, crop=None):
    pass

def plot3d(img_3d):
    l = 30
    l = img_3d.shape[1] 
    vol = np.zeros((l, l, l))
    pts = (l * np.random.rand(3, 15)).astype(np.int)
    vol[tuple(indices for indices in pts)] = 1
    from scipy import ndimage
    vol = ndimage.gaussian_filter(vol, 4)
    vol /= vol.max()
    img_3d = vol

    X, Y, Z = np.mgrid[:l, :l, :l]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=img_3d.flatten(),
        isomin=0.2,
        isomax=0.7,
        opacity=0.1,
        surface_count=25,
        ))
    fig.show()

def view3d(cls, pid, scandid):
    folder = '%s/%s/%d/%d' % (settings.DATA_PATH_ROOT, cls, pid, scandid)
    files = utils.get_filelist(folder)
    abs_files = [os.path.join(folder, f) for f in files]
    imgs = [imageio.imread(os.path.join(folder, f), as_gray=True) for f in files]
    imgs = [np.expand_dims(img, axis=0) for img in imgs]
    img_3d = np.concatenate(imgs)
    print('image shape: ', imgs[0].shape)
    print('3d image shape: ', img_3d.shape)
    print('files: ', files)
    #plt.imshow(imgs[0][0,...], cmap=plt.get_cmap('gray'))
    #plt.show()
    plot3d(img_3d)



if __name__ == '__main__':
    view3d('CP', 4, 3508)

