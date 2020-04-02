# 
# Implementation of RASL: 
#   Robust Alignment of images by Sparse and Low-rank decomposition
#   http://perception.csl.illinois.edu/matrix-rank/rasl.html
#

import numpy as np
from geometry import *
from matplotlib.pyplot import *
from matrix_approx import low_rank_approximation, soft_threshold
from visualization.video_writer import composite_movie
import logging

#logger = logging.getLogger()
#logging.basicConfig(level=logging.DEBUG)
#logging.setLevel(logging.DEBUG)
#logger = logging.getLogger(__name__)

debug_large=False

#@profile
def rasl_inner(D, J=None, L_fixed=None, L_fixed_inv=None, rank=10, lam=0.1,
        tol=1e-3, maxIter=500,
        power=3, use_qr=False, low_rank_method='SVT'): 
    '''Transformed Robust PCA.
    Fits a model of the form: X o \tau = L + S
    '''
    Y = D.copy() #XXX: neccessary?
    norm_fro = np.linalg.norm(D)
    # Approximate the two norm
    norm_two = np.linalg.norm(Y, 2)
    #norm_two = low_rank_approximation(Y, 'BRP', rank=1, power=10)
    norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lam
    dual_norm = np.maximum(norm_two, norm_inf)
    Y /= dual_norm
    obj_v = (D*Y).sum()
    m, n = D.shape
    A_dual = np.zeros_like(D)
    E_dual = np.zeros_like(D)
    dt_dual = [ [] for i in xrange(n)] 
    dt_dual_matrix = np.zeros_like(D)
    rho = 1.25
    mu = rho / norm_two
    d_norm = norm_fro
    iter = 0
    converged = False
    if L_fixed is not None and L_fixed_inv is None:
        L_fixed_inv = np.linalg.pinv(L_fixed)
    while not converged:
        iter += 1
        temp_T = D + dt_dual_matrix - E_dual + (1/mu) * Y
        if L_fixed is not None:
            Vhat = np.dot(L_fixed_inv, temp_T)
            A_dual = np.dot(L_fixed, Vhat)
        else:
            #U, V = low_rank_approximation(temp_T, 'NN', rank=rank, lam=1/mu, power=power)
            if low_rank_method == 'SVT':
                U, V = low_rank_approximation(temp_T, 'SVT', lam=1/mu)
            elif low_rank_method =='BRPSVT':
                U, V = low_rank_approximation(temp_T, 'BRPSVT', rank=rank, lam=1/mu, power=power)
            else:
                U, V = low_rank_approximation(temp_T, 'BRP', rank=rank, power=power)
            A_dual = np.dot(U, V.T)
        temp_T = D + dt_dual_matrix - A_dual + (1/mu) * Y
        E_dual = soft_threshold(temp_T, lam / mu)
        temp_T -= E_dual + dt_dual_matrix
        if J is not None:
            for i in xrange(n):
                if use_qr:
                    dt_dual[i] = -np.dot(J[i].T, temp_T[:, i])
                    dt_dual_matrix[:, i] = np.dot(J[i] , dt_dual[i])
                #iJ = np.linalg.inv(np.dot(J[i].T, J[i]))
                else:
                    dt_dual[i] = -np.linalg.lstsq(J[i], temp_T[:, i])[0]
                #dt_dual[i] = -np.dot(J[i].T, temp_T[:, i])
                #dt_dual[i] = np.dot(iJ, dt_dual[i])
                dt_dual_matrix[:, i] = np.dot(J[i] , dt_dual[i])
        Z = D + dt_dual_matrix - A_dual - E_dual
        Y += mu * Z
        obj_v = (D*Y).sum()
        mu = mu * rho
        stoppingCriterion = np.linalg.norm(Z) / d_norm
        #logging.debug('\tstoppingCriterion=%g'%stoppingCriterion)
        #print '\tstoppingCriterion=%g'%stoppingCriterion
        if stoppingCriterion <= tol or iter >= maxIter:
            converged = True
    return A_dual, E_dual, dt_dual, iter, Y

def center_transforms(transforms):
    mu = np.array([tfm.params[:3] for tfm in transforms]).mean(0)
    for transform in transforms:
        transform.params[:3] -= mu

#@profile
from util.volume import subsample_volume

def rasl_multiscale(data, vol_shape, scales=[1], L_fixed=None, border=[8,8,8],
        transform_type='PartialEuclidean', transforms=None, **kwargs):
    if data.ndim == 1:
        data = data.reshape(1,-1)
    if transforms is None:
        N = data.shape[0]
        transform_class = TRANSFORMS[transform_type]
        transforms = [transform_class() for i in xrange(N)]
    # Sort scales in descending order
    scales = np.sort(scales)[::-1]
    vmax = None
    for subsample_factor in scales:
        #print 'Subsampling by a factor of', subsample_factor
        # Rescale transforms for appropriate scale
        for tfm in transforms:
            tfm.params[:3] /= subsample_factor
        if subsample_factor == 1:
            data_sub, vol_shape_sub = data, vol_shape
        else:
            data_sub, vol_shape_sub = subsample_volume(data, vol_shape,
                    subsample_factor, blur=False)
        border = (np.array(border) / float(subsample_factor)).astype(int)
        if L_fixed is not None:
            vol_shape_2 = np.array(vol_shape).copy()
            vol_shape_2[1:] /= scales[-1]
            subsamp = subsample_factor/scales[-1]
            if subsamp == 1:
                LL = L_fixed
            else:
                LL = subsample_volume(L_fixed.T, vol_shape_2, subsample_factor/scales[-1])[0].T
        else:
            LL = None
        A, D, L, E, transforms = rasl_outer(data_sub, vol_shape_sub, 
            L_fixed=LL, transforms=transforms, border=border, **kwargs)
        import copy
        old_transforms = copy.deepcopy(transforms)
        for tfm in transforms:
            tfm.params[:3] *= subsample_factor
        if L_fixed is None: 
            from visualization.video_writer import composite_movie
            from visualization.factors import implot
            from matplotlib.pyplot import *
            TD_sub = transform_dataset(data_sub, vol_shape_sub, old_transforms)
            new_transforms = copy.deepcopy(old_transforms)
            for tfm in new_transforms:
                tfm.params[:3] *= subsample_factor
            TD = transform_dataset(data, vol_shape, transforms)
            std = TD.std(0)
            print 'subsamp=%d, mean(std)=%g' % (subsample_factor, std.mean())
            figure(1);clf()
            if vmax is None:
                vmax = std.max()
            implot(std, vol_shape,vmin=0.0, vmax=vmax)
            savefig('/home/poolio/aligned_keyframes_var_%d.png' % subsample_factor)
            composite_movie([data,TD],vol_shape,'/home/poolio/aligned_keyframes_%d.mp4'%subsample_factor)
            composite_movie([data_sub,TD_sub],vol_shape_sub,'/home/poolio/aligned_keyframes_%d_sub.mp4'%subsample_factor)
    if L_fixed is None:
        return A,D,L,E,transforms
    else:
        return transforms

def rasl_outer(data, vol_shape, transform_type='PartialEuclidean', 
        lambda_inner=1.0, max_iter=30, low_rank_method='SVT',
        inner_rank = 30, normalize=True, sigma=None, use_qr=False, border=2,
        L_fixed=None, debug_movies=False,
        outer_rank=None, lambda_outer=1.0, inner_tol=1e-5, transforms=None,
        tol=1e-7):
    """Align a dataset using the RASL algorithm. 

    Args:
        data: nframes x nvoxels
        vol_shape: tuple containing (sizez, sizey, sizex) of volume
        transform_type: Type of transformation to fit. One of:
            'Translation': Translations in x,y,z
            'Euclidean': Translations and rotations in x,y,z
            'PartialEuclidean': Translations in x,y,z and rotations in x,y only
        lambda_inner: sparsity penalty for aligned frames
        max_iter: maximum number of iterations
        low_rank_method: low-rank matrix approximation technique. either
            'SVT': singular value threshold (slower, more accurate)
            'BRPSVT': ranodmized SVD then SVT (faster, less accurate)
        inner_rank: maximum rank of low-rank component for BRPSVT
        normalize: Scale frames to have unit norm. Default is true, as it
            prevents a trivial solutions with large translations.
        sigma: Standard deviation of isotropic Gaussian filter to apply to 
            volumes before computing gradients with Sobel filter. Defaults
            to None for no smoothing.
        L_fixed (optional): nvoxels x nfactors Fixed basis for low-rank component.
            When set, RASL will not update L at each iteration, and instead
            will solve a least squares problem to identify L.
        use_qr: Orthogonalize Jacobian at each step, which helps with numerical
            stability. Default is true.
        border: Size of border to create around transformed images. Non-zero
            values help to reduce problems with edge artifacts.
        debug_movies: Output a debug movie at each iteration.
    """
    if transform_type not in TRANSFORMS:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    if low_rank_method not in ['SVT', 'BRPSVT', 'BRP']:
        raise ValueError('Unknown low rank method: %s' % low_rank_method)
    if not normalize:
        mean_norm = np.mean(np.sqrt(np.sum(data**2,axis=1)))
        data /= mean_norm
    converged = False
    prevObj = np.inf
    iterNum = 0
    # If we're passed a vector to align, reshape it to a matrix
    if data.ndim == 1:
        data = data.reshape(1,-1)
    # Precompute inverse to speed up low-rank estimation
    L_fixed_inv = np.linalg.pinv(L_fixed) if L_fixed is not None else None

    # Initialize transform vectors
    N = data.shape[0] 
    if transforms is None:
        transform_class = TRANSFORMS[transform_type]
        transforms = [transform_class() for i in xrange(N)]
    J = [[] for i in xrange(N)]
    Q = [[] for i in xrange(N)]
    R = [[] for i in xrange(N)]
    D = np.empty_like(data)
    lambda_inner /= np.sqrt(D.shape[1])
    lambda_outer /= np.sqrt(D.shape[1])
    objs = []
    #XXX: set parameter
    iterStart = 3 
    param_history = []
    # Align data!
    while not converged:
        if L_fixed is None:
            # Center frames to prevent drift
            center_transforms(transforms)
        params_old = [tfm.params.copy() for tfm in transforms]
        #param_history.append(np.array(params_old))
        #print np.array(params_old).mean(0)
        if outer_rank is not None and iterNum > iterStart:
            TD = transform_dataset(data, vol_shape, transforms)
            inverse_transforms = map(lambda x: x.inverse(), transforms)
            if iterNum == 0:
                res = TD
            else:
                res = TD - L.T
            invL = transform_dataset(res, vol_shape, inverse_transforms)
            Lo, Eo, _, _, _ = rasl_inner(invL, rank=inner_rank, lam=lambda_outer, L_fixed=L_fixed)
            A  = Lo
        else:
            A = np.zeros_like(data)
        res = data - A
        for i in xrange(N):
            D[i], J[i] = image_jacobian(vector_to_volume(res[i], vol_shape),
                    transforms[i], sigma, normalize=normalize, border=border)
            # Perform QR decomposition to orthogonalize Jacobian
            if use_qr:
                Q[i], R[i] = np.linalg.qr(J[i])
        jac_arg = Q if use_qr else J
        # Solve transformed robust PCA problem
        L, E, delta_params, numIterInnerEach, Y = rasl_inner(D.T, jac_arg,
                rank=inner_rank, lam=lambda_inner, use_qr=use_qr,
                L_fixed=L_fixed, L_fixed_inv=L_fixed_inv,
                low_rank_method=low_rank_method, tol=inner_tol)
        # Compute new transformations
        for i in xrange(N):
            if use_qr:
                delta_params[i] = np.linalg.lstsq(R[i], delta_params[i])[0]
                #delta_xi[i] = np.dot(np.linalg.inv(R[i]), delta_xi[i])
            transforms[i].params += delta_params[i]
        # Calculate new objective function
        U, s, V = np.linalg.svd(L,full_matrices=False)
        curObj = np.linalg.norm(s,1) + lambda_inner*np.linalg.norm(E.ravel(),1) 
        #logging.debug('curObj=%g'%curObj)
        if debug_movies:# and debug_large):
            TD = transform_dataset(data, vol_shape, transforms)
            print 'iter=', iterNum, ' curObj=',curObj
            composite_movie([data,TD,TD-A,TD-A-(TD-A).min(0),A,L.T,E.T], vol_shape, 'iter%d.mp4'%iterNum,min_sub=False)
            np.savez('transforms_iter%d'%iterNum,
                    transforms=transforms, L=L)
        dtfm = [np.linalg.norm(tfm.params-param_old) for tfm, param_old in
                zip(transforms, params_old)]
       # print 'MEAN PARAM CHANGE', np.mean(dtfm)
        #objs.append(curObj)
        objs.append(np.mean(dtfm))
        if iterNum >max_iter or np.mean(dtfm) < tol:
            print 'Converged'
            converged = True
        #print objs
        iterNum += 1
    #np.save('/home/poolio/param_history.npy', param_history)
    #if L_fixed is not None:
    #    return transforms
    #else:
    return A, D, L.T, E.T, transforms

def generate_synthetic_dataset(vec, vol_shape, T):
    data = np.tile(vec, (T, 1))
    transform_class = TRANSFORMS['PartialEuclidean']
    srange = 2
    shifts = 2*srange*(np.random.rand(T, 3) - 0.5)
    shifts[:, 2] = 0.0
    shifts -= shifts[0]
    rotations = np.random.rand(T)/10.
    transforms = [transform_class(shift=shifts[i],rotation=rotations[i]) for i in xrange(T)]
    center_transforms(transforms)
    data = transform_dataset(data, vol_shape, transforms)
    inverse_transforms = map(lambda x: x.inverse(), transforms)
    #center_transforms(inverse_transforms)
    return data, transforms, inverse_transforms

if __name__ == '__main__':
    subsamp = 1
    if 1:
        f = np.load('/data/share/facedata.npz')
        data = f['data']
        vec = data[0]
        vol_shape = f['vol_shape']
        T = 100
        data, true_transforms, true_inv_transforms = generate_synthetic_dataset(vec, vol_shape, T)
        data_subsamp = data[::subsamp].copy()
    else:
        from util.io import load_data
        datareader = load_data('/data/share/maxproj_align.h5')
        data = datareader.data
        vol_shape = datareader.vol_shape
        data_subsamp = data[::subsamp]

    simulate = False

    #1/0
    import time
    tic = time.time()
    A, D, L, E, transforms = rasl_outer(data_subsamp, vol_shape, outer_rank=None,
            inner_rank=40, max_iter=10,sigma=None,lambda_inner=1.0,
            lambda_outer=0.1, normalize=True, L_fixed=None,
            transform_type='PartialEuclidean',
            border=5,low_rank_method='SVT', debug_movies=True)
    center_transforms(transforms)
    Dt = transform_dataset(data_subsamp, vol_shape, transforms)
    composite_movie([data_subsamp,Dt], vol_shape, '/tmp/out_d1.mp4')
    params = np.array([t.params for t in transforms])
    if simulate:
        tparams = np.array([t.params for t in true_inv_transforms])
        print 'Batch MAD=',np.mean(np.abs(tparams[::subsamp]-params))
    #params -= params[0]
    U, s, V = np.linalg.svd(L, full_matrices=False)
    idx = np.where(s > 1e-7)[0]
    #L_fixed = np.dot(U[:, idx], np.diag(s[idx]))
    L_fixed = np.dot(np.diag(s[idx]), V[idx, :]).T.copy()
    L_fixed = L.T.copy()
    transforms2 = rasl_outer(data, vol_shape, outer_rank=None,
            inner_rank=40, max_iter=10,sigma=None,lambda_inner=1.0,
            lambda_outer=0.1, normalize=True, L_fixed=L_fixed,
            transform_type='Translation', border=5)
    print 'Alignment complete. Took %d seconds.' % (time.time() - tic)
    center_transforms(transforms2)
    offparams = np.array([t.params for t in transforms2])
    if simulate:
        print 'Online MAD=',np.mean(np.abs(tparams-offparams))
    #Dt = transform_dataset(data, vol_shape, transforms)
    Dt2 = transform_dataset(data, vol_shape, transforms2)
    composite_movie([data_subsamp,Dt,Dt2[::subsamp]], vol_shape, '/tmp/out_d2.mp4')
