import numpy as np
import glob
from matplotlib.pyplot import *
from util.volume import volume_to_vector

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#files = glob.glob( '/Users/poole/Dropbox/RASL_Code/data/maxproj_images_decimated_nominsub/*')
files = glob.glob( './maxproj_images_decimated/*')

import Image

data = None
for i,f in enumerate(files):
    imrgb = imread(f).astype(float)
    im = np.array(Image.open(f))
    #im = im[50:-50, 50:-50]
    #im /= np.linalg.norm(im.ravel())
    #im[:10,:] = 0.0
    #im[:,:25] = 0.0
    if data is None:
        data = np.zeros((len(files), im.size))
        vol_shape = im.shape + (1,)
        vol_shape = vol_shape[::-1]
    data[i] = volume_to_vector(im.reshape(vol_shape[::-1]))
    #data[i] = im.ravel(order='f')
#vol_shape = im.shape

np.savez('/Users/poole/brainmin3d.npz', data=data,vol_shape=vol_shape)
#np.savez('/Users/poole/brainmin.npz', data=data,vol_shape=vol_shape)



