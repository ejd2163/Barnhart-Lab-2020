import sys,os
import numpy as np
# Path to pyrasl folder
pyrasl_path = '.'
sys.path.append(pyrasl_path)

from util.data_conversion import generate_volume
from util.io import load_data, save_data
from alignment.rasl import RASL
from alignment.geometry import transform_dataset
from util.viz import composite_movie

# Path to input tif file
input_tif = '../miroKK_main.tif'
# Output diretory
output_dir = '../results'
# Name of converted HDF5 file
base_name = os.path.splitext(os.path.basename(input_tif))[0] 
output_h5 = os.path.join(output_dir, base_name + '.h5')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert tif -> h5. Only needs to be run once.
if not os.path.exists(output_h5):
    generate_volume(input_tif, output_h5)

# Load dataset 
datareader = load_data(output_h5)
vol_shape = datareader.vol_shape
data_orig = datareader.data

# Gamma correction, set gamma=1.0 for no correction.
gamma = 0.6
data = data_orig.astype(float)/data_orig.max()
data = data ** gamma

# Remove original data to save memory
del data_orig

# Number of keyframes used to find template
n_keyframes = 20
data_keyframes = data[::len(data)/n_keyframes]

# Align keyframes using RASL. See docstring for details about parameters.
debug_movies = False
aligner = RASL(
            use_qr=True,
            scales=[(4,4,1),(2,2,1)],
            max_iter=20,
            lam=2.0,
            normalize=True,
            transform_type='Euclidean',
            border=[20,20,2],
            low_rank_method='SVT',
            debug_directory=output_dir,
            debug_movies=debug_movies,
            compute_objective=True,
            verbose=True)

# Find low-rank template on keyframes
L,E,transforms = aligner.fit(data_keyframes, vol_shape)

# Align all frames in parallel
# Here we specify a smaller number of iterations as we're
# fixing the template and typically converge rapidly.
aligner.max_iter = 8
final_transforms = aligner.parallel_align(data, n_jobs=-1,chunks=10)#,scales=[4,3,2])

# Save transforms
transform_fn = os.path.join(output_dir, base_name + '_transforms.npy')
np.save(transform_fn, final_transforms)

# Transform original dataset
# order specifies the interpolation order
# For now higher-order interpolation does
# some weird things so we stick with linear.
del data
datareader = load_data(output_h5)
vol_shape = datareader.vol_shape
data = datareader.data

aligned_fn = os.path.join(output_dir, base_name + '_aligned.h5')
aligned_data = transform_dataset(data, vol_shape, final_transforms, order=1)
save_data(aligned_data, vol_shape, aligned_fn)

# Generate output movie comparing unaligned (left) and aligned (right)
# You will need ffmpeg for this to work
aligned_movie_fn = os.path.join(output_dir, base_name + '_movie.mp4')
composite_movie([data, aligned_data], vol_shape, aligned_movie_fn)

