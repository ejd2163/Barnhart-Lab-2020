#
# Script to convert a set of images to an h5 file
# Example usage:
#   python images_to_h5.py "input_directory/\*.png" test.h5
#
import numpy as np
import glob
import h5py
import Image
from argparse import ArgumentParser
from util.volume import volume_to_vector

parser = ArgumentParser()
parser.add_argument('input_path', help="Path for a glob, e.g. input_directory/*.png")
parser.add_argument('output_filename', help="Output h5 filename")
options = parser.parse_args()
# Ordered list of files
files = glob.glob(options.input_path)
ntimesteps = len(files)

data = None
for i, f in enumerate(files):
    im = np.array(Image.open(f))
    if data is None:
        # Initialize h5 dataset on disk
        nvoxels = im.size
        dims = im.shape + (1,)
        vol_shape = dims[::-1]
        out= h5py.File(options.output_filename, 'w')
        out.create_dataset('vol_shape', data=vol_shape)
        data= out.create_dataset('timeseries_volume', (ntimesteps, nvoxels), dtype=im.dtype)
    data[i] = volume_to_vector(im.reshape(dims))
out.close()

# Example visualization
from visualization.factors import implot
from matplotlib.pyplot import show
from util.io import load_data
datareader = load_data(options.output_filename)
implot(datareader[0], datareader.vol_shape)
show()
