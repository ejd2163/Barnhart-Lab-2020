import os
import h5py
import numpy as np
from util.volume import grid_to_idx, vector_to_volume, volume_to_vector
from scipy.ndimage.filters import gaussian_filter1d
from util.console_output import simple_progress_bar


class DataReader(object):
    def load(self):
        pass
    def read(self, tslice=None, vslice=None):
        self.load()
        tslice = tslice if tslice is not None else slice(None)
        vslice = vslice if vslice is not None else slice(None)
        return self.data[tslice, vslice]
    def serialize(self):
        pass
    def __getitem__(self, index):
        return self.read(index, None)


class MemoryDataReader(DataReader):
    def __init__(self, data, vol_shape):
        self.data = data
        self.vol_shape = vol_shape
        self.ntimesteps = data.shape[0]
        self.nvoxels = data.shape[1]


class FileDataReader(DataReader):
    def __init__(self, filename, in_memory=False):
        self.filename = filename
        self.data = None
        self.load()
    def load(self):
        if self.data is not None:
            return
        self.file = h5py.File(self.filename, 'r')
        self.vol_shape = self.file['vol_shape'][...]
        self.data = self.file['timeseries_volume']
        self.ntimesteps, self.nvoxels = self.file['timeseries_volume'].shape

    def serialize(self):
        # If file is open before forking, HDF5 is unpredictable.
        # https://groups.google.com/forum/#!msg/h5py/bJVtWdFtZQM/MpgAphp4gBMJ
        if self.data is not None:
            self.data = None
            self.file.close()
            self.file = None

    def __del__(self):
        self.file.close()


def save_data(data, vol_shape, filename):
    output_file = h5py.File(filename, 'w')
    output_file.create_dataset('vol_shape', data=vol_shape)
    data = output_file.create_dataset('timeseries_volume', data=data)
    output_file.close()


def load_data(input_filename, time_window=None, crops=None,
                 big_data=False, subsample_time=1,
                 output_filename=None, chunks=100, min_sub=False,
                 max_proj=False, smooth=False):
    '''Load dataset into DataReader object. Can be read into memory using .read()

    :param input_filename: Input h5 filename
    :param time_window: tuple of [start_time, end_time]
    :param subsample_time: Subsampling factor in time. Note this does not do smoothing
                           before subsampling.
    :param crops: crop volumes according to list of 3 tuples:
                    [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    :param big_data: Write output to h5 file if cropped, and return FileDataReader
    :param output_filename: Output file to write if using big_data (optional)
    :param chunks: Read dataset in this many different chunks.
    :param min_sub: Min subtract dataset
    :param smooth: Whether to average frames before subsampling.
    '''
    if big_data and output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + '_cropped.h5'
    input_data = FileDataReader(input_filename)
    vol_shape = input_data.vol_shape
    time_window = time_window if time_window is not None else [0, input_data.ntimesteps]
    time_window = np.array(time_window)
    if time_window[1] == -1:
        time_window[1] = input_data.ntimesteps
    time_list = np.arange(time_window[0], time_window[1], subsample_time)
    #XXX: ben throwing away data!
    ntimesteps = len(time_list)
    do_crop = crops is not None and np.any(crops > 0)
    do_time = ntimesteps != input_data.ntimesteps
    if min_sub:
        if 'vol_min' not in input_data.file:
            raise ValueError('Run update_timeseries_stats on %s' % input_filename)
        vol_min = input_data.file['vol_min'][...]
    if not (do_crop or do_time or min_sub or max_proj):
        if big_data:
            return input_data
        else: 
            return MemoryDataReader(input_data.read(), input_data.vol_shape)
    if do_crop:
        idx, new_vol_shape = grid_to_idx(vol_shape, crops[0], crops[1], crops[2])
    else:
        nocrops=True
        idx = slice(None)
        new_vol_shape = vol_shape
    if max_proj:
        new_vol_shape = np.array(new_vol_shape)
        new_vol_shape_full = new_vol_shape.copy()
        new_vol_shape[0] = 1


    nvoxels = np.prod(new_vol_shape)
    # Allocate memory or file
    if big_data:
        print('Writing output to ', output_filename)
        output_file = h5py.File(output_filename, 'w')
        output_file.create_dataset('vol_shape', data=new_vol_shape)
        data = output_file.create_dataset('timeseries_volume',
            (ntimesteps, nvoxels), dtype=input_data.data.dtype)
    else:
        data = np.zeros((ntimesteps, nvoxels), dtype=input_data.data.dtype)
    # Read dataset into memory or file
    #output_tchunks = np.array_split(np.arange(ntimesteps), chunks)

    if subsample_time !=1 and smooth:
        ntimesteps_last = (time_window[1] - time_window[0])  % (subsample_time * chunks)
        tlist_full = np.arange(time_window[0], time_window[1])
        input_tchunks = np.array_split(tlist_full[:len(tlist_full)-ntimesteps_last], chunks)
        if ntimesteps_last > 0:
            input_tchunks.append(tlist_full[-ntimesteps_last:])
            chunks += 1
    else:
        input_tchunks = np.array_split(time_list, chunks)
    tidx = 0
    sigma = subsample_time / 4.0
    for i, input_tchunk in enumerate(input_tchunks):
        simple_progress_bar(i, chunks)
        if input_tchunk.size: # array_split can have empty arrays
            # h5 only supports one indexing vector
            dchunk = input_data.read(input_tchunk, slice(None))
            if dchunk.ndim == 1:
                dchunk = dchunk.reshape(1, -1)
            dchunk = dchunk[:, idx]
            if subsample_time != 1 and smooth:
                dchunk = gaussian_filter1d(dchunk, sigma=sigma, axis=0)[::subsample_time]
            #if preproc is not None:
            #    dchunk = preproc(dchunk)
            if min_sub:
                dchunk -= vol_min[idx]
            output_tchunk = np.arange(tidx, tidx + dchunk.shape[0])
            tidx += dchunk.shape[0]
            if max_proj:
                for t in xrange(dchunk.shape[0]):
                    data[output_tchunk[t]] = volume_to_vector(vector_to_volume(dchunk[t], new_vol_shape_full).max(2)[:, :, np.newaxis])
            else:
                data[output_tchunk, :] = dchunk
    # Return data as DataReader objects
    if big_data:
        output_file.close()
        out_data = FileDataReader(output_filename)
    else:
        out_data = MemoryDataReader(data, new_vol_shape)
    return out_data
