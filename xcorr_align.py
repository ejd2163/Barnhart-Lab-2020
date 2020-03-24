from scipy.signal import convolve
from scipy.signal import fftconvolve
#fftconvolve = convolve
from scipy.signal import correlate
from scipy.ndimage.interpolation import zoom, shift
from scipy.ndimage import interpolation, filters
from util.volume import vector_to_volume, volume_to_vector, apply_volume, chunked_map
import numpy as np
from joblib import Parallel, delayed
import operator

def normalize(x):
    y =  (x - x.mean())
    return y / y.std()

def compute_keyframes(datareader, chunks=100):
    # Compute mean volume for each chunk
    chunk_means = chunked_map(datareader.data,
                              lambda x: x.mean(0),
                              chunks=chunks)
    #keyframes = np.array(reduce(operator.add, chunk_means)) 
    return np.vstack(chunk_means)#keyframes

def align_data(data, vol_shape,  base=None, n_jobs=1,shifts=None, sequential=False):
    #mean_vec = normalize(data.std(0))
    if base is None:
        mean = vector_to_volume(data.mean(0), vol_shape)
    else:
        mean = vector_to_volume(base, vol_shape)
    vol = vector_to_volume(data, vol_shape)
    if True:
        vol = filters.gaussian_filter(vol, 1.0)

    loc = align(mean,vol , shifts)
    return loc

    locs = np.zeros((data.shape[0], data.ndim))
    if sequential or n_jobs == 1:
        for t in xrange(data.shape[0]):
            print t
            frame = vector_to_volume(data[t, :], vol_shape)
            locs[t, :] = align(mean, frame, shifts)
            if sequential:
                mean = interpolation.shift(frame, shifts[t,:])
    else:
#        mean = vector_to_volume(normalize(data[t,:]), vol_shape)
        output=Parallel(n_jobs=n_jobs)(delayed(align)(mean,
            vector_to_volume(normalize(data[t,:]), vol_shape), shifts) for t in range(data.shape[0])) 
        #for t in xrange(data.shape[0]):
           # new_data[t, :] = volume_to_vector(output[t])
        locs = np.vstack(output)
    return locs

def align_all_data(data, vol_shape,  base=None, n_jobs=1,shifts=None, sequential=False):
    #mean_vec = normalize(data.std(0))
        #sdata = data.std(0)
    def preproc(x):
        return vector_to_volume(x, vol_shape)
    #    return filters.gaussian_filter(vector_to_volume(x, vol_shape), 1.0)
        #return filters.gaussian_gradient_magnitude(vector_to_volume(x, vol_shape), 1.0)
    if base is None:
        mean = preproc(data.mean(0))
    else:
        mean = preproc(base)

        #return vector_to_volume(x, vol_shape)
    if sequential:
        locs = np.zeros((data.shape[1], len(vol_shape)))
        for t in xrange(data.shape[0]):
            print t
            frame = preproc(data[t,:])
            #frame = vector_to_volume(data[t, :], vol_shape)
            #if True:
            #    frame = filters.gaussian_filter(frame, 1.0)
            locs[t, :] = align(mean, frame, shifts)
            if sequential:
                mean = interpolation.shift(frame, locs[t,:])
    else:
#        mean = vector_to_volume(normalize(data[t,:]), vol_shape)
        output=Parallel(n_jobs=n_jobs)(delayed(align)(mean,
            preproc(data[t,:]), shifts) for t in range(data.shape[0])) 
        #for t in xrange(data.shape[0]):
           # new_data[t, :] = volume_to_vector(output[t])
        locs = np.vstack(output)

    return locs


def zero_border(vol, border=3):
    vol[:border, :, :] = 0
    vol[:, :border, :] = 0
    vol[:, :, :border] = 0
    vol[-border:, :, :] = 0
    vol[:, -border:, :] = 0
    vol[:, :, -border:] = 0
    return vol

def transform_frame_fnc(loc, border=3):
    return lambda x: interpolation.shfit(x, loc)

from rpca import remove_artifacts
#def transform_frame(frame, vol_shape, loc, V, Vinv):
#    vol = vector_to_volume(frame, vol_shape)
#    newvol = zero_border(shift(remove_artifacts(vol,V,Vinv), loc), border=3)
#    vec = volume_to_vector(newvol)


def transform_frame(frame, vol_shape, loc, V, Vinv):
    if V is not None:
        frame = remove_artifacts(frame, V, Vinv)
    vol = vector_to_volume(frame, vol_shape)
    newvol = zero_border(shift(vol, loc), border=3)
    vec = volume_to_vector(newvol)
    return vec

def transform_data(datareader, locs, vol_shape, border=3, n_jobs=-1,
        output_fn=None):
    assert data.shape[0] == locs.shape[0]
    fnc = lambda x, loc: zero_border(interpolation.shift(x, loc),border=3)
    output = apply_volume(data, vol_shape, fnc, targs=locs)[0]
    #chunked_map_ooc(datareader, fnc
    #output=Parallel(n_jobs=n_jobs)(delayed(interpolation.shift) (
    #        vector_to_volume(data[t,:], vol_shape), loc) for t,loc in
    #        enumerate(locs)) 
    #return np.vstack(output)
    return output

def align_dataset(datareader, chunks=100, n_jobs=-1, shifts=(3,3,2)):
    keyframes = compute_keyframes(datareader, chunks)
    offsets = chunked_map_ooc(datareader, align_data, targs=keyframes, n_jobs=n_jobs,
            chunks=chunks, shifts=shifts)
    return offsets

    aligned_datareader = chunked_map_ooc(datarader, transform_data,
            output_filename=output_filename, targs=offsets, n_jobs=n_jobs, chunks=chunks)
    return aligned_datareader

def align_data_blocks(data, vol_shape, nblocks, n_jobs=-1, shifts=None):
    tchunks = np.array_split(np.arange(data.shape[0]), nblocks)
    means = np.vstack([data[chunk, :].mean(0) for chunk in tchunks])
    print 'Aligning means'
    mean_locs = align_data(means, vol_shape, n_jobs=n_jobs, shifts=shifts)

    aligned_means = transform_data(means, mean_locs, vol_shape)

    print 'Aligning individual frames'
    output = []
    for t in xrange(nblocks):
        chunk = tchunks[t]
        mean = aligned_means[t, :]
        #, mean in zip(range(nblocks),tchunks, aligned_means):
        print t
        out = align_data(data[chunk,:],vol_shape,shifts=shifts,n_jobs=n_jobs,sequential=False,base=mean)
        output.append(out)

    #output = Parallel(n_jobs=n_jobs)(delayed(align_data)(data[chunk, :],
    #    vol_shape, shifts=shifts, n_jobs=1, sequential=True,base=mean) for
    #    chunk,mean in
    #    zip(tchunks, aligned_means))

    output = np.vstack(output)
    return output

    #for chunk in tchunks:
    #    out = align_data(data[chunk, :], vol_shape, shifts=shifts,sequential=True) 

def myconvolve(x, y):
    '''fftconvolve. x must be bigger than y'''
    shape = np.array(x.shape) - np.array(y.shape) + 1
    out = np.zeros(shape)
    for i in xrange(out.size):
        ix,iy,iz = np.unravel_index(i, shape)
        sub = x[ix:ix+y.shape[0], iy:iy+y.shape[1], iz:iz+y.shape[2]]
        res = (sub*y).mean() / sub.std()
        out[ix,iy,iz] = res
    return out


def align(base, frame, shifts=None):
    print '.'
    input_zoom_factor = 3
    base = interpolation.zoom(base, input_zoom_factor)
    base = normalize(base)
    frame = interpolation.zoom(frame, input_zoom_factor)
    frame = normalize(frame)
    if shifts is None:
        shifts = (5,) * base.ndim
    #XXX: assume base and frame are normalized?
    # Flip frame
    #XXX: investigate weird odd/even-ness
    #mins = [shift-1 if shift > 0 else None for shift in shifts]
    
    #slices = [slice(-shift-1, shift-1, -1) if shift > 0 else slice(-1,None,-1) for shift in shifts]
    slices = [slice(shift, -shift-1,1) for shift in shifts]
    
    #xcorr = fftconvolve(base, frame[slices], 'valid')
    #flipped_base = normalize(base[slices])
    print 'convstatrt'
    xcorr = myconvolve(frame, base[slices])
    print 'convend'
    #xcorr = 1./flipped_base.size * fftconvolve(frame, flipped_base, 'valid')
    #o = np.ones_like(flipped_base)
    #ocorr = 1./o.size * fftconvolve(frame**2, o, 'valid')
    #omean = 1./o.size * fftconvolve(frame, o, 'valid')
    #std = np.sqrt(ocorr - omean**2)
    #xcorr_orig = xcorr.copy()
#    xcorr /= std

    # Compute location of peak
    # Upsample to get more exact peak
    if 0:
        maxidx = np.argmax(xcorr)
        coord = np.unravel_index(maxidx, xcorr.shape)
        from scipy.interpolate import griddata
        points = np.vstack(np.mgrid[:xcorr.shape[0], :xcorr.shape[1],:xcorr.shape[2]])
        print '('
        steps = 10
        slices = [slice(np.linspace(c-1, c+1, steps)) for c in coord]
        grid = np.mgrid[slices]
        output = griddata(points, xcorr, grid, method='cubic') 
        print ')'
        1/0

    zoom_factor = 2
    fine_xcorr = interpolation.zoom(xcorr, zoom_factor)
    maxidx = np.argmax(fine_xcorr)
    coord = np.unravel_index(maxidx, fine_xcorr.shape)
    loc = np.array([np.linspace(-shift, shift, nshift)[idx] for shift, nshift, idx in
        zip(shifts, fine_xcorr.shape, coord)])
    loc = -loc
    print loc
    #print coord
    # Shift input frame to get result
    #return xcorr,fine_xcorr, base[slices][::-1,::-1,::-1], ocorr, omean, xcorr_orig,std
    #return interpolation.shift(frame, loc)
    return loc/input_zoom_factor
    #return


