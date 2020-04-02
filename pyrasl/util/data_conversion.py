"""
Utilities for converting other data formats (Matlab, .tif) into 
standard .h5 format
"""
import numpy as np
import glob, os
import h5py
from util.imageio import load_image
from util.volume import volumes_to_vectors
from util.console_output import simple_progress_bar

def generate_volume(dirname, output_name=None, maxt=None):
    path, ext = os.path.splitext(dirname)
    if ext == '.tif':
        if output_name is None:
            output_name = path + '.h5'
        tif_files = [dirname]
    else:
        tif_files = sorted(glob.glob(os.path.join(dirname,'*.tif')))
        if maxt:
            tif_files = tif_files[:maxt]
        if output_name is None:
            output_name = os.path.join(dirname, 'volume.h5')
    output_h5 = h5py.File(output_name, 'w')
    initialized = False
    # Assumes that there is a single stack containing 2d images 
    if len(tif_files) == 1:
        f = tif_files[0]
        print('Warning: only found one tif file. Assuming it contains a temporal stack of 2d images')
        vol = load_image(f)
        print('\t--> Reading: ', f, '   Dimensions: ', vol.shape)
        vol_shape = np.array([vol.shape[2], vol.shape[1], vol.shape[0]])
        print("vol_shape= ", vol_shape)

        data_shape = (vol.shape[3], np.prod(vol_shape[:3]))
        output_volume = output_h5.create_dataset('timeseries_volume', data_shape, 'f')
        for t in xrange(vol.shape[3]):
            output_volume[t, :] = np.reshape(np.reshape(vol[:,:,:,t], vol_shape[::-1]), np.prod(vol_shape), order='f')
    else:
        for t, f in enumerate(tif_files):
            vol = load_image(f)
            print('\t--> Reading: ', f, '   Dimensions: ', vol.shape)
            vol_row = np.reshape(vol, np.prod(vol.shape), order='f')
            vol_shape = np.array([vol.shape[2], vol.shape[1], vol.shape[0]])

            # If this is the first pass through the loop, we need
            # to allocate the large matrix into which we will
            # place volumes as individual rows.
            if not initialized:
                data_shape = (len(tif_files), vol_row.shape[0])
                output_volume = output_h5.create_dataset('timeseries_volume', data_shape, 'f')
                initialized = True
            output_volume[t,:] = vol_row
    print('Saving volume to', output_h5)
    output_h5.create_dataset('vol_shape', data=vol_shape[...])
    output_h5.close()

def leong_matlab_to_h5(input_filename, output_filename=None):
    if output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + '.h5'
    f = h5py.File(input_filename,'r')
    dsets = [f[ref[0]] for ref in f['imagingStruct']['stackRaw']]
    mat_shape = dsets[0].shape
    T = np.sum([dset.shape[1] for dset in dsets])
    vol_shape = (1, mat_shape[3], mat_shape[2])
    func_data = np.zeros((T, np.prod(vol_shape)),dtype=dsets[0].dtype)
    struct_data = np.zeros((T, np.prod(vol_shape)),dtype=dsets[0].dtype)
    output_file = h5py.File(output_filename, 'w')
    output_file.create_dataset('vol_shape', data=vol_shape)
    func_data = output_file.create_dataset('timeseries_volume', (T, np.prod(vol_shape)), dtype=dsets[0].dtype)
    struct_data = output_file.create_dataset('structural_volume', (T, np.prod(vol_shape)), dtype=dsets[0].dtype)
    idx = 0
    for dset in dsets[:]:
        datum = np.transpose(dset[...], [1,2,3,0])
        func_datum = volumes_to_vectors(datum[..., 1])
        func_data[idx:idx+dset.shape[1], :] = func_datum
        struct_datum = volumes_to_vectors(datum[..., 0])
        struct_data[idx:idx+dset.shape[1], :] = struct_datum
    f.close()
    output_file.close()

def matlabv7_to_h5(input_filename, data_name, output_filename=None):
    if output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + '.h5'
    f = h5py.File(input_filename,'r')
    matdata = f[data_name]
    fout = h5py.File(output_filename, 'w')
    img_shape = matdata.shape[1:][::-1]
    vol_shape = np.array((1,) + img_shape)
    data_shape = (matdata.shape[0], np.prod(vol_shape))
    fout.create_dataset('vol_shape', data=vol_shape)
    output_volume = fout.create_dataset('timeseries_volume', data_shape, 'f')
    for t in xrange(matdata.shape[0]):
        simple_progress_bar(t, matdata.shape[0])
        output_volume[t] =  np.reshape(np.reshape(matdata[t], vol_shape[::-1]), np.prod(vol_shape), order='f')
    fout.close()

def subsample_volume(input_name, factor=2, output_name=None):
    '''Subsample volume'''
    # Read input
    input_h5 = h5py.File(input_name, 'r')
    input_volume = input_h5['timeseries_volume']
    vol_shape = input_h5['vol_shape'][...]
    # Setup output
    if output_name is None:
        output_name = '%s_subsampled_%dx.h5' % (os.path.splitext(input_name)[0], factor)
    if os.path.exists(output_name):
        return output_name
    output_h5 = h5py.File(output_name, 'w')
    output_volume = None
    for t in xrange(input_volume.shape[0]):
        x, new_vol_shape = volume.subsample_volume(input_volume[t,:], vol_shape, factor)
        if output_volume is None:
            data_shape = (input_volume.shape[0], x.shape[0])
            output_volume = output_h5.create_dataset('timeseries_volume', data_shape, 'f')
        output_volume[t, :] = x
        simple_progress_bar(t, input_volume.shape[0])
    print('Saving volume to', output_h5)
    output_h5.create_dataset('vol_shape', data=new_vol_shape[...])
    output_h5.flush()
    output_h5.close()
    return output_name
