# Utilities for manipulationg light field volumes
import numpy as np
import sys
from scipy.ndimage import filters
from util.console_output import simple_progress_bar

class ZeroVolumeError(Exception):
    pass

def check_zeros(data):
    if np.allclose([data.min(), data.max()], 0.0):
        raise ZeroVolumeError()

def vector_to_volume(vec, vol_shape):
    '''Convert vector to volume with shape (vol_shape[2], vol_shape[1], vol_shape[0])'''
    return np.reshape(vec, vol_shape[::-1], order='F').copy()


def volume_to_vector(vol):
    '''Convert volume to Fortran-ordered vector'''
    return np.reshape(vol, -1, order='F').copy()


def vectors_to_volumes(vecs, vol_shape):
    '''Convert num_vectors x vector_length matrix to num_vectors x volume shape tensor'''
    return np.reshape(vecs, (-1, vol_shape[2], vol_shape[1], vol_shape[0]), order='F').copy()


def volumes_to_vectors(vols):
    '''Convert volumes to num_volumes x voxels_per_volume matrix'''
    return np.reshape(vols, (-1, np.prod(vols.shape[1:])), order='F').copy()


def idx_to_coord(idx, vol_shape):
    '''Convert vector indices into 3-tuple containing volume coordinates.

    :param idx: scalar indices into vectorized volume
    :param vol_shape: tuple of volume shape Z,Y,X 
    '''
    return np.unravel_index(idx, vol_shape[::-1], order='F')

def coord_to_idx(coord, vol_shape):
    '''Convert volume coordinates to indices into vectorized volumes.

    :param coord: 3-tuple containing lists of x, y, and z coordinates
    :param vol_shape: tuple of volume shape Z,Y,X 
    '''
    # Flatten arrays so we return a flat list of indices
    if not np.isscalar(coord[0]) and coord[0].ndim > 1:
        coord = [coord_dim.flatten() for coord_dim in coord]
    return np.ravel_multi_index(coord, vol_shape[::-1], order='F')


def grid_to_idx(vol_shape, rangex, rangey, rangez):
    '''Convert range of coordinates into indices into vectorized volume.

    :param vol_shape: tuple of volume shape Z,Y,X 
    :param rangex: range of x coordinates, [rangex[0], rangex[1])
    :param rangey: 
    :param rangez:
    '''
    dims = vol_shape[::-1]
    if rangex[1] < 0:
        rangex[1] = vol_shape[2] + rangex[1] + 1
    if rangey[1] < 0:
        rangey[1] = vol_shape[1] + rangey[1] + 1
    if rangez[1] < 0:
        rangez[1] = vol_shape[0] + rangez[1] + 1
    # Generate flat indices from meshgrid
    coords = np.mgrid[rangex[0]:rangex[1], rangey[0]:rangey[1], rangez[0]:rangez[1]]
    idx = coord_to_idx(coords, vol_shape)
    new_vol_shape = (rangez[1] - rangez[0], rangey[1] - rangey[0], rangex[1] - rangex[0])
    # Reorder meshgrid indices so they correspond to flattened Fortran-ordered indices.
    # This is necessary so that we can reshape the flattened vectors back into volumes using
    # the existing vector_to_volume code. TODO(ben): make this less hacky 
    new_coords = np.mgrid[0:new_vol_shape[2], 0:new_vol_shape[1], 0:new_vol_shape[0]]
    new_idx = coord_to_idx(new_coords, new_vol_shape)
    sorted_idx = np.argsort(new_idx)
    idx = idx[sorted_idx]
    return idx, new_vol_shape

def sparse_vector(vol_shape, idx, coeff=1.0, dtype=np.float32):
    vec = np.zeros(np.prod(vol_shape), dtype=dtype)
    vec[idx] = coeff
    return vec

def apply_volume(data, vol_shape, fn, targs=None,**kwargs):
    '''Apply a function to the volume at each timestep

    :param data: dataset
    :param vol_shape: tuple of volume shape (Z,Y,X)
    :param fn: Function to apply to volumes. Takes as input a single volume
               and returns a volume of the same shape.
    '''
    if data.ndim == 1:
        data = data.reshape(1,-1)
    if targs is not None:
        assert len(targs) == data.shape[0]
    out = None
    args = []
    for t in xrange(data.shape[0]):
        vol = vector_to_volume(data[t,:], vol_shape)
        if targs is not None:
            args = [targs[t]]
        outt = fn(vol, *args, **kwargs)
        res = volume_to_vector(outt)
        if out is None:
            new_vol_shape = outt.shape[::-1]
            out = np.empty((data.shape[0], res.shape[0]), dtype=res.dtype)
        out[t, :] = res
    return out, new_vol_shape

def apply_vector(data, fn, chunks=100, **kwargs):
    '''Apply a function to the volume at each timestep

    :param data: dataset
    :param vol_shape: tuple of volume shape (Z,Y,X)
    :param fn: Function to apply to volumes. Takes as input a single volume
               and returns a volume of the same shape.
    '''
    out = None
    ntimesteps = data.shape[0]
    input_xchunks = np.array_split(np.arange(data.shape[1]), chunks)
    for xchunk in input_xchunks:
        print('.')
        vec =data[:,xchunk]
        res = fn(vec, **kwargs)
        if out is None:
            out = np.empty((res.shape[0], data.shape[1]), dtype=res.dtype)
        out[:, xchunk] = res
    return out


def subsample_dataset(datareader, factor=100, chunks=10): 
    datareader.ntimesteps -= datareader.ntimesteps % (factor * chunks)
    #ntimesteps = int(np.floor(datareader.ntimesteps / float(factor)))
    #ntimesteps / chunks
    #datareader.ntimesteps = int(ntimesteps * factor)
    sigma = factor / 4
    #chunks = np.minimum(chunks, ntimesteps)
    #chunks = chunks / ntimesteps
    preproc = lambda x: filters.gaussian_filter1d(x, sigma=sigma, axis=0)[::factor]
    out = chunked_map_ooc(datareader, fn=None, preproc=preproc, chunks=chunks, subsamp_factor=factor, vol_shape_arg=False, aout=True)
    return out

def chunked_map_ooc(datareader, fn, targs=None, chunk_args=None,chunks=20,
        output_filename=None, n_jobs=-1, aout=True, preproc=None,
        vol_shape_arg=True, overwite=True, transpose=False,subsamp_factor=1, **kwargs):

    '''Apply a function to chunks of the vectorized datset.

    :param data: dataset
    :param fn: Function to apply to chunks of dataset.
    :param targs: Argument for each frame of dataset
    :param chunk_args: Argument for each temporal chunk
    :param chunks: Number of chunks to split data into
    :param n_jobs: Number of jobs to use
    :param aout: Array output. Reduces the output of `fn` to an array 
                 either in memory or to h5 file output_filename if not None.
                 If aout is True, then we return a DataReader, otherwise
                 we return a list containing the function outputs.
    :param output_filename: h5 file to write output if aout=True
    :param **kwargs:  Additional kwargs to be passed to `fn`
    '''

    # Using numpy within multiprocessing segfaults on Mac.
    # See: http://mail.python.org/pipermail/python-ideas/2012-November/017932.html
    if sys.platform == 'darwin':
        n_jobs = 1
    from io import FileDataReader, MemoryDataReader
    out = None
    ntimesteps = datareader.ntimesteps
    nvoxels = datareader.nvoxels
    ntimesteps_out = int(np.floor(datareader.ntimesteps / float(subsamp_factor)))
    vol_shape = datareader.vol_shape
    if transpose:
        input_tchunks = np.array_split(np.arange(nvoxels), chunks)
    else:
        input_tchunks = np.array_split(np.arange(ntimesteps), chunks)

    args = []
    carg = []
    tidx = 0
    for t, tchunk in enumerate(input_tchunks):
        simple_progress_bar(t, len(input_tchunks))
        if tchunk.size:
            # Select appropriate parameters
            if targs is not None:
                args = targs[tchunk]
            elif chunk_args is not None:
                args = [chunk_args[t]]*len(tchunk)
            else:
                args = None
            # Read chunk of data
            if transpose:
                data = datareader.read(None, tchunk)
            else:
                data = datareader.read(tchunk, None)
            if preproc:
                if vol_shape_arg:
                    data, vol_shape= preproc(data, datareader.vol_shape)
                else:
                    data = preproc(data).astype(data.dtype)
            # Compute fnc in parallel over chunks of data
            from joblib import Parallel, delayed
            #TODO(ben): clean up/simplify
            if fn is not None:
                if vol_shape_arg:
                    if args is not None:
                        tout = Parallel(n_jobs=n_jobs)(delayed(fn) 
                                (datum, vol_shape, arg, **kwargs)
                                for datum, arg in zip(data, args))
                    else:
                        tout = Parallel(n_jobs=n_jobs)(delayed(fn) 
                                (datum, vol_shape, **kwargs)
                                for datum in data)
                else:
                    if args is not None:
                        tout = Parallel(n_jobs=n_jobs)(delayed(fn) 
                                (datum, arg, **kwargs)
                                for datum, arg in zip(data, args))
                    else:
                        tout = Parallel(n_jobs=n_jobs)(delayed(fn) 
                                (datum, **kwargs)
                                for datum in data)
            else:
                tout = [data]
            if aout:
                # Compress output into array
                tout = np.vstack(tout)
                if out is None:
                    if transpose:
                        shape = (tout.shape[0], nvoxels)
                    else:
                        shape = (ntimesteps_out,) + tout.shape[1:]
                    if output_filename is None:
                        out = np.zeros(shape, dtype=tout.dtype)
                    else:
                        import h5py
                        f = h5py.File(output_filename, 'w')
                        f.create_dataset('vol_shape', data=vol_shape)
                        out = f.create_dataset('timeseries_volume',
                                shape,
                                dtype=tout.dtype)
                tchunk_out = np.arange(tidx, tidx + tout.shape[0])
                if transpose:
                    out[:, tchunk_out] = tout
                else:
                    out[tchunk_out, :] = tout
                tidx += tout.shape[0]
            else:
                # Keep output in a big list
                if out is None:
                    out = []
                out.extend(tout)
        simple_progress_bar(t, chunks)
    if aout:
        if output_filename is not None:
            f.close()
            outreader = FileDataReader(output_filename)
        else:
            outreader = MemoryDataReader(out, vol_shape)
        return outreader
    else:
        return out


def chunked_map(data, fn, chunks=100, output_filename=None, **kwargs):
    '''Apply a function to chunks of the vectorized datset.

    :param data: dataset
    :param fn: Function to apply to chunks of dataset.
    :param chunsk: Number of chunks to split data into
    '''
    out = []
    ntimesteps = data.shape[0]
    input_tchunks = np.array_split(np.arange(ntimesteps), chunks)
    for tchunk in input_tchunks:
        if tchunk.size:
            tout = fn(data[tchunk, :], **kwargs)
            out.append(tout)
    return out

def chunked_compute(data, mapfn, reducefn=None, chunks=100):
    '''Map-reduce over chunks of a dataset.

    :param mapfn: Takes a chunk of data and creates an output
    :param reducefn: Reduces list of outputs into single value.
                     If reducefn is None, re-apply mapfn to output.
     '''
    outs = chunked_map(data, mapfn, chunks)
    if reducefn is None:
        out = mapfn(np.vstack(outs))
    else:
        out = reducefn(outs)
    return out

def chunked_mean(data, chunks=100):
    '''Compute mean of data one chunk at a time.'''
    mapfn = lambda x: x.sum(0)
    reducefn = lambda x: np.vstack(x).sum(0) / data.shape[0]
    return chunked_compute(data, mapfn, reducefn, chunks)

def chunked_std(data, mean=None, chunks=100):
    '''Compute std of data.'''
    if mean is None:
        mean = chunked_mean(data, chunks)
    mapfn = lambda x: ((x-mean)**2).sum(0)
    reducefn = lambda y:  np.sqrt(np.vstack(y).sum(0) / data.shape[0])
    return chunked_compute(data, mapfn, reducefn, chunks)

def subsample_volume(timeseries, vol_shape, factor = 2, blur=True):
    '''Subsample volume along each dimension.

    :param timeseries: dataset
    :param vol_shape: tuple of volume shape (Z,Y,X)
    :param factor: Keep every factor voxels in each dimension. A factor of 2
                   corresponds to an 8x decrease in the number of voxels.
    '''
#    if factor == 1:
#        return timeseries.copy(), vol_shape
    if np.isscalar(factor):
        factor = np.ones(len(vol_shape)) * factor
    factor = np.array(factor)
    #row = timeseries[0,:]
    #vol = vector_to_volume(row, vol_shape)

    # Compute appropriate smoothing and downsampling factors
    dims = vol_shape[::-1]
    # Only do subsampling if volume is large enough
    do_subsample = dims >= 2 * factor
    slices = tuple(np.array([slice(0, dims[d], factor[d]) if do_subsample[d] else slice(None) for d in xrange(3)]))
    # Compute smoothing factors
    if blur:
        sigmas = np.where(do_subsample, factor/2.0, 1.0)
    X = None
    for t in range(timeseries.shape[0]):
    #    if t%10 ==0:
    #        print '\t--> processing ', t,'/', X.shape[0], ' time slices'
        row = timeseries[t,:]
        vol = vector_to_volume(row, vol_shape)
        #vol = filters.uniform_filter(vol, size=factor)
        if blur:
            vol = filters.gaussian_filter(vol, sigma=sigmas)
        vol = vol[slices]
        if X is None:
            new_vol_shape = np.array([vol.shape[2], vol.shape[1], vol.shape[0]])
            X = np.zeros((timeseries.shape[0], np.prod(new_vol_shape)))
        X[t,:] = volume_to_vector(vol)
    return X, new_vol_shape


def volume_adjacency(vol_shape, power=1):
    '''Volume adjacency matrix. Nonzero elements indicate adjacent voxels.
    Note that we dont flip vol_shape because vectorized volume is Fortran-ordered.

    :param vol_shape: tuple of volume shape (Z,Y,X)
    :param power: Raise adjacency matrix to this power to get higher-order connectivity.
    '''
    from sklearn.feature_extraction.image import grid_to_graph
    adjacency = grid_to_graph(*vol_shape).tocsr()
    return adjacency**power


def volume_laplacian(vol_shape):
    '''Volume Laplacian.'''
    adjacency = volume_adjacency(vol_shape)
    degree_diag = np.array(adjacency.sum(axis=1)).T[0]
    import scipy.sparse as sps
    degree = sps.spdiags(degree_diag, np.array([0]), len(degree_diag),len(degree_diag))
    L = sps.csr_matrix((degree - adjacency), dtype=np.float32)
    return L


def project_vec(vec, vol_shape, **kwargs):
    '''Create max projection image from vectorized volume

    :param vol: 1d vectorized volume
    :param axis: axis to project across or -1 for orthogonal views
    '''
    return project_volume(vector_to_volume(vec, vol_shape), **kwargs)


def project_volume(vol, border=1, axis=-1, xy_spacing = None, z_spacing = None):
    '''Create max projection image from volume.

    :param vol: 3d volume
    :param axis: axis to project across or -1 for orthogonal views
    :param xy_spacing, z_spacing:  if these are set, the aspect ratio
                     of the side projections are adjusted to match so that
                     xz and yz pixel resolution is isotropic
    '''
    wx,wy,wz = vol.shape
    if axis == -1 or axis == -2:
        if xy_spacing != None and z_spacing != None:
            from scipy.misc import imresize
            aspect = float(z_spacing) / xy_spacing

            xz_projection = np.flipud(np.nanmax(vol, axis=0).T)
            output_size = (int(xz_projection.shape[0] * aspect), xz_projection.shape[1])
            xz_projection = imresize(xz_projection, output_size, mode = 'F')  # mode is 'F' for floating point

            yz_projection = np.fliplr(np.nanmax(vol, axis=1))
            output_size = (yz_projection.shape[0], int(yz_projection.shape[1]*aspect))
            yz_projection = imresize(yz_projection, output_size, mode = 'F') 

            wx,wy,wz = vol.shape                
        
        else:
            xz_projection = np.flipud(np.nanmax(vol, axis=0).T)
            yz_projection = np.fliplr(np.nanmax(vol, axis=1))

        # Create the projections and populate them.
        I = np.nan*np.zeros((wx + xz_projection.shape[0] + border + 1,
                             wy + yz_projection.shape[1] + border + 1), dtype=float)
        if axis == -1:
            I[:wx, :wy] = np.nanmax(vol, axis=2)
            I[wx+border:-1, :wy] = xz_projection 
            I[:wx, wy+border:-1] = yz_projection
        elif axis == -2:
            I[-wx:, -wy:] = np.flipud(np.nanmax(vol, axis=2))
            I[:-wx-border-1, -wy:] = xz_projection
            I[-wx:, :-wy-border-1] = np.flipud(yz_projection)
            I = I.T
    else:
        if xy_spacing != None and z_spacing != None:
            raise NotImplementedError("Aspect ratio correction is only implemented for orthogonal projections (axis = -1).")
        elif axis == 0:
            I = np.flipud(np.nanmax(vol, axis=0).T)
        elif axis == 1:
            I = np.fliplr(np.nanmax(vol, axis=1))
        else:
            I = np.nanmax(vol, axis=2)

    return I


def safe_norm(X, eps=1e-9):
    import ncreduce
    '''Compute norm of rows of X, adding epsilon to zero entries'''
    normX = np.sqrt(X.shape[0]) * ncreduce.std(X, 0)
    normX[normX == 0.0] = 1e-9
    return normX

def zero_border(vol, border=1, cval=0.0):
    """Zero-out boundary voxels from a volume.

    Args:
        vol: 3d volume
        border: scalar or tuple containing borders for each dimension
        cval: constant value to replace boundary voxels with, default is 0.0
    """
    dims = np.array(vol.shape)
    if np.isscalar(border):
        border = border * np.ones(3, dtype=int)
        # Don't apply border to singleton dimensions.
        border[dims == 1] = 0
    border = np.array(border, dtype=int)
    if np.any(dims - border <= 0):
        raise ValueError('Border %s exceeds volume shape %s.' %
                (str(border), str(dims)) )
    for dim, bval in enumerate(border):
        if bval > 0:
            slices = [slice(bval) if d == dim else slice(None) for d in xrange(3)]
            vol[slices] = cval
            slices[dim] = slice(-bval, None)
            vol[slices] = cval
    return vol

def pad_volume(vol, padding=[2,2,2], cval=0.0):
    dims = np.array(vol.shape)
    dims += 2*np.array(padding)
    vol_out = np.zeros(dims, dtype=vol.dtype)
    slices = [slice(p, d - p) for d,p in zip(dims, padding)]
    vol_out[slices] = vol
    return vol_out


