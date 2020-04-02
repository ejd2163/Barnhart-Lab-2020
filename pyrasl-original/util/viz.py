from volume import project_vec
from pipeffmpeg import create_h264_mp4
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import * 

def nice_colorbar(ax, obj, labels=True):
    """Nicely scaled colorbar"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    c=colorbar(obj, cax=cax)
    if not labels:
        cax.set_yticklabels([])
        cax.set_xticklabels([])
    sca(ax)

def implot(vec, shape, ax=None, vmin=None, vmax=None,
        labels=True,colorbar=True, cmap=None, **kwargs):
    if ax is None:
        ax = gca()
    im = ax.imshow(project_vec(vec, shape, **kwargs), interpolation='nearest',
            vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        nice_colorbar(ax, im, labels)
    ax.grid(color='w')
    # Hide axis labels for now
    if not labels:
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
    return ax

def composite_movie(vecs, vol_shape, outfn, min_sub=False, rate=30,
        gamma=1.0,axis=-1, verbose=False):
    """Generate an mp4 file containing multiple data movies.
    
    :param vecs: list of ntime x nvoxels arrays
    """
    if min_sub:
        vecs = [vec - vec.min(0) for vec in vecs]
    vecs = [np.sign(vec)*(np.abs(vec)/vec.max())**gamma for vec in vecs]
    mins = [vec.min() for vec in vecs]
    maxs = [vec.max() - vmin for vec, vmin in zip(vecs, mins)]
    video_writer = None
    for t in xrange(vecs[0].shape[0]):
        ims = []
        for vec, vmin, vmax in zip(vecs, mins, maxs):
            im = (project_vec(vec[t,:], vol_shape,axis=axis) - vmin) / vmax
            ims.append(im)
        frame = np.hstack(ims)
        frame = np.array(255*frame,dtype=np.uint8)
        if video_writer is None:
            video_writer = create_h264_mp4(outfn, frame.shape, fps=rate,
                    verbose=verbose)
        video_writer.writeframe(frame)
    video_writer.close()

def display_local_video(fn):
    import io
    import base64
    from IPython.display import HTML
    video = io.open(fn, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))
