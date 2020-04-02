# 
# Implementation of RASL: 
#   Robust Alignment of images by Sparse and Low-rank decomposition
#   http://perception.csl.illinois.edu/matrix-rank/rasl.html
#

import numpy as np
import logging
import copy
import os
from alignment.geometry import *
from matplotlib.pyplot import *
from alignment.matrix_approx import low_rank_approximation, soft_threshold, tensor_admm
from util.viz  import composite_movie
from util.volume import subsample_volume, vectors_to_volumes, volumes_to_vectors
from util.volume import chunked_map_ooc
from util.io import MemoryDataReader, DataReader

#logger = logging.getLogger()
#logging.basicConfig(level=logging.DEBUG)
#logging.setLevel(logging.DEBUG)
#logger = logging.getLogger(__name__)

debug_large=False


from scipy.optimize import fmin_l_bfgs_b

def f(x, A, b):
    return np.linalg.norm(np.dot(A,x)-b)**2/2.0

def fprime(x, A, b):
    return -np.dot(A.T, b) + np.dot(A.T, np.dot(A, x))

def constrained_ls(A, b, bounds):
    x0 = np.linalg.lstsq(A, b)[0]
    out = fmin_l_bfgs_b(f, x0, fprime, args=(A,b), bounds=bounds)
    x = out[0]
    return x

    chunked_map_ooc(datareader, align_fn, rasl=self)

def default_align_fn(data, vol_shape, idx, rasl):
    """Helper function for multiprocessing to align a single frame."""
    _, _, transforms = rasl.align(data, vol_shape)
    return transforms[0]


INITIAL_TRANSFORMS = []
transform_class = TRANSFORMS['PartialEuclidean']
def robust_align_fn(data, vol_shape, idx, rasl, scales=None):
    """Helper function for multiprocessing to align a single frame."""
    errors = []
    tfms = []
    INITIAL_TRANSFORMS = [transform_class(rotation=rot) for rot in np.linspace(0,2*np.pi, 10)]
    for tfm in INITIAL_TRANSFORMS: 
        _, E, transforms = rasl.align(data, vol_shape, initial_transforms=[tfm], scales=scales)
        errors.append(np.linalg.norm(E.ravel()))
        tfms.append(transforms[0])
    best_tfm = tfms[np.argmin(errors)]
    if scales is not None:
        _, _, transforms = rasl.align(data, vol_shape, initial_transforms=[best_tfm])
        best_tfm = transforms[0]
    return best_tfm

class RASL(object):
    #def rasl_multiscale(data, vol_shape, scales=[1], L_fixed=None, border=[8,8,8],
    #    transform_type='PartialEuclidean', transforms=None, **kwargs):

    def __init__(self, transform_type='PartialEuclidean', scales=[1], lam=0.1, low_rank_method='SVT', use_qr=False, border=[8,8,8],
        debug_movies=False, debug_directory='.', rank=50, max_iter=10,
        sigma=None, normalize=True, L=None, tensor_weighting=None,power=4,
        compute_objective=False, verbose=False):
        """Align a dataset using the RASL algorithm.

        Args:
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
        # Sort scales in descending order
        scale_ordering = np.argsort([np.sum(scale) for scale in scales])[::-1]
        self.scales = np.array(scales)[scale_ordering]
        self.border = np.array(border)
        if transform_type not in TRANSFORMS:
            raise ValueError('Unknown transform_type: %s' % transform_type)
        #if low_rank_method not in ['SVT', 'BRPSVT', 'BRP']:
        #    raise ValueError('Unknown low rank method: %s' % low_rank_method)
        self.transform_type = transform_type
        self.low_rank_method = low_rank_method
        self.lam = lam
        self.use_qr = use_qr
        self.debug_movies = debug_movies
        self.debug_directory = debug_directory
        if self.debug_movies and not os.path.exists(self.debug_directory):
            os.makedirs(self.debug_directory)
        self.rank = rank
        self.max_iter = max_iter
        self.sigma = sigma
        self.normalize = normalize
        self.L = None
        self.tensor_weighting = tensor_weighting
        self.tensor = self.tensor_weighting is not None
        self.verbose = verbose
        #if L is not None:
        #    self.set_low_rank(L)
        self.compute_objective = compute_objective

    def set_low_rank(self, L, vol_shape):
        #XXX: assumes scale of L is the finest scale
        dtest, vol_shape_L = subsample_volume(np.zeros((1, np.prod(vol_shape))), vol_shape, self.scales[-1])
        self.L = []
#        if len(L) != len(self.scales):
#            raise ValueError('Number of low-rank matrices not equal to number of scales')
        Ls = [L]
        for scale in self.scales[:-1][::-1]:
            #XXX: scale/self.scales[-1] could be non-integer....
            Lsub, Lsubshape = subsample_volume(L, vol_shape_L, scale/self.scales[-1])
            Ls.append(Lsub)
        Ls = Ls[::-1]
        self.L = [Li.T for Li in Ls]
        self.L_inv = [np.linalg.pinv(Li) for Li in self.L]
        #self.L_hat = [np.dot(Li, Linv) for Li, Linv in zip(self.L, self.L_inv)]


    # def _fixed_inner_loop(self, D, J, lam=0.1, tol=1e-5, maxIter=500):
    #     # Scaling of lam?
    #     Y = D.copy()
    #     norm_two = np.linalg.norm(D,2)
    #     Y /= norm_two
    #     rho = 1.25
    #     mu = rho / norm_two
    #     m,n = D.shape
    #     converged = False
    #     while not converged:
    #         residual = D + dt_dual_matrix - E_dual + (1/mu) * Y
    #         # Solve for low rank component
    #         A_dual = np.dot(self.L_hat, residual)
    #         # Solve for delta tau
    #         if J is not None:
    #             for i in xrange(n):
    #                 if use_qr:
                        



    def _inner_loop(self, D, J=None, L_fixed=False, L_fixed_inv=None, rank=10, lam=0.1,
            tol=1e-3, maxIter=20,
            power=3, use_qr=False, low_rank_method='SVT'): 
        """ransformed Robust PCA.
        Fits a model of the form: X o \tau = L + S
        """
        Y = D.copy() #XXX: neccessary?
        # Rescale lambda
        lam = self.lam/np.sqrt(D.shape[0])#np.prod(D.shape))
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
        Y = np.zeros_like(Y)
        if 0:#self.L_fixed:
            print(Y.shape)
            #print Y.mean(1)
            print(dict(mu=mu))
            mu = 0.21

        dparam = J[0].shape[1]
        prange = 1.
        bounds = [(-prange, prange) for i in xrange(dparam)]
        while not converged:
            iter += 1
            # Update low-rank component
            residual = D + dt_dual_matrix - E_dual + (1/mu) * Y
            if self.L_fixed:
                A_dual = np.dot(self.L[self.current_scale], np.dot(self.L_inv[self.current_scale], residual))
            else:
                if self.tensor:
                    # Pull out T x d1 x d2 x d3 tensor
                    residual = vectors_to_volumes(residual.T, self.vol_shape)
                    A_dual = tensor_admm(residual, lam=1/mu, gammas=self.tensor_weighting, eta=0.1, tol=1e-3, maxiter=30, verbose=False)
                    # Reshape back to D x T matrix
                    A_dual = volumes_to_vectors(A_dual).T
                else:
                    if self.low_rank_method == 'SVT':
                        U, V = low_rank_approximation(residual, 'SVT', lam=1/mu)
                    elif self.low_rank_method =='BRPSVT':
                        U, V = low_rank_approximation(residual, 'BRPSVT', rank=self.rank, lam=1/mu, power=power)
                    else:
                        U, V = low_rank_approximation(residual, 'BRP', rank=self.rank, power=power)
                    A_dual = np.dot(U, V.T)
            # Update delta tau, the change in transformations
            residual = D - E_dual - A_dual + (1/mu) * Y
            if J is not None:
                for i in xrange(n):
                    if 1:
                        if self.use_qr:
                            dt_dual[i] = -np.dot(J[i].T, residual[:, i])
                        else:
                            dt_dual[i] = -np.linalg.lstsq(J[i], residual[:, i])[0]
                    else:
                        dt_dual[i] = - constrained_ls(J[i], residual[:, i], bounds)

                    dt_dual_matrix[:, i] = np.dot(J[i] , dt_dual[i])
            # Update sparse component
            residual = D + dt_dual_matrix - A_dual + (1/mu) * Y
            E_dual = soft_threshold(residual, lam / mu)
            # Update 
            Z = D + dt_dual_matrix - A_dual - E_dual
            Y += mu * Z
            obj_v = (D*Y).sum()
            mu = mu * rho
            stoppingCriterion = np.linalg.norm(Z) / d_norm
            #logging.debug('\tstoppingCriterion=%g'%stoppingCriterion)
            if 0:#self.L_fixed:
                print('\t%d: f=%g'%(iter, stoppingCriterion))
            if stoppingCriterion <= tol or iter >= maxIter:
            #    print 'iter=',iter, ' crit=', stoppingCriterion
                converged = True
        return A_dual, E_dual, dt_dual, iter, Y



    def fit(self, data, vol_shape, initial_transforms=None, L_fixed=False, scales=None):
        """Align dataset using RASL.

        Args:
            data: nframes x nvoxels
            vol_shape: tuple containing (sizez, sizey, sizex) of volume
            transforms: list of transformation objects
            border: Size of border to create around transformed images. Non-zero
                values help to reduce problems with edge artifacts.
            tol: 
        Returns:
            L: Low-rank matrix
            E: sparse matrix
            transforms: list of transformation objects
        """

        if scales is None or not L_fixed:
            scales = self.scales
        self.L_fixed = L_fixed
        #self.vol_shape = vol_shape
        # Handle case of aligned single image
        if data.ndim == 1:
            data = data.reshape(1,-1)
        # Store tensor shape
        #if self.tensor:
            #dims = np.hstack((data.shape[0], vol_shape[::-1]))
            #self.tensor_shape = (np.prod(dims[self.tensor_split[0]]), 
            #                     np.prod(dims[self.tensor_split[1]]))
        # Initialize transforms
        if initial_transforms is None:
            N = data.shape[0]
            transform_class = TRANSFORMS[self.transform_type]
            transforms = [transform_class() for i in xrange(N)]
        else:
            assert len(initial_transforms) == len(data)
            transforms = initial_transforms
        Ls = []
        # Compute the fit for each scale
        for idx, scale in enumerate(scales):
            self.current_scale = idx
            # Resize transforms and data to appropriate scale
            for tfm in transforms: tfm.resize(1./scale)
            data_sub, vol_shape_sub = subsample_volume(data, vol_shape, scale, blur=True)
            self.vol_shape = vol_shape_sub
            self.grid = GridTransformer(self.vol_shape[::-1])
            border = (self.border / scale).astype(int)
            # Run RASL on scaled data and parameters 
            L, E, transforms = self._outer_loop(data_sub, vol_shape_sub, transforms=transforms, border=border)
            # Reduce dimensionality of L here...
            U, s, V = np.linalg.svd(L, full_matrices=False)
            idx = np.where(s > 1e-12)[0]
            L_small = np.dot(np.diag(s[idx]), V[idx, :])
#            Ls.append(L.mean(0).reshape(1,-1))
            Ls.append(L_small)
            # Scale up transforms
            for tfm in transforms: tfm.resize(scale)
            # Optionally save out debug movies
            if self.debug_movies:
                movie_fn = os.path.join(self.debug_directory, 'scale%d_final.mp4' % self.current_scale)
                # Transform dataset with learned transforms
                data_aligned = transform_dataset(data, vol_shape, transforms)
                composite_movie([data, data_aligned], vol_shape, movie_fn)
        # Store low-rank component for future runs
        if not L_fixed:
            self.set_low_rank(Ls[-1], vol_shape)
        # Set vol_shape to appropriate input vol_shape
        self.vol_shape = vol_shape
        return L,E,transforms

    def align(self, data, vol_shape, initial_transforms=None, scales=None):
        """Align a dataset using the already computed fixed low-rank component."""
        if self.L is None:
            raise ValueError("No low-rank component. Run fit() or set_low_rank first.")
        return self.fit(data, vol_shape, initial_transforms, L_fixed=True, scales=scales)

    def parallel_align(self, data,  multiple_inits=False, **kwargs):
        """Align each frame of a dataset in parallel.

        Args:
            data: nframes x nvoxels numpy array, or
                  DataReader object that supports reading from disk or memory.
            multiple_inits: WIP: attempt many initial rotations for robustness
            **kwargs: additional arguments for alignment function
        """
        # Convert to datareader objects
        if not isinstance(data, DataReader):
            data = MemoryDataReader(data, self.vol_shape)
        align_fn = robust_align_fn if multiple_inits else default_align_fn
        self.debug_movies = False
        self.verbose = False
        self.compute_objective = False
        aaa=chunked_map_ooc(data,
                align_fn,targs=np.arange(data.ntimesteps), rasl=self, aout=False, **kwargs)
        return aaa

    def _outer_loop(self, data, vol_shape, transforms, border, tol=1e-7):
        """Align a dataset using the RASL algorithm. 

        Args:
            data: nframes x nvoxels
            vol_shape: tuple containing (sizez, sizey, sizex) of volume
            transforms: list of transformation objects
            border: Size of border to create around transformed images. Non-zero
                values help to reduce problems with edge artifacts.
            tol: 
        Returns:
            L: Low-rank matrix
            E: sparse matrix
            transforms: list of transformation objects
        """
        
        if not self.normalize:
            mean_norm = np.mean(np.sqrt(np.sum(data**2,axis=1)))
            data /= mean_norm
        converged = False
        prevObj = np.inf
        iterNum = 0
        # If we're passed a vector to align, reshape it to a matrix
        if data.ndim == 1:
            data = data.reshape(1,-1)

        N = data.shape[0] 
        J = [[] for i in xrange(N)]
        Q = [[] for i in xrange(N)]
        R = [[] for i in xrange(N)]
        D = np.empty_like(data)
        lambda_inner = self.lam/np.sqrt(np.prod(D.shape[1]))
        #lambda_outer /= np.sqrt(D.shape[1])
        objs = []
        param_history = []
        # Align data!
        while not converged:
           # if L_fixed is None:
            # Center frames to prevent drift
            if not self.L_fixed:
#                center_transforms(transforms, rotation=not self.tensor)
                center_transforms(transforms, rotation=True)
            params_old = [tfm.params.copy() for tfm in transforms]
            old_transforms = copy.deepcopy(transforms)
            #param_history.append(np.array(params_old))
            #print np.array(params_old).mean(0)
            for i in xrange(N):
                D[i], J[i] = image_jacobian(vector_to_volume(data[i], vol_shape),
                        transforms[i], self.grid, self.sigma, normalize=normalize, border=border)
                # Perform QR decomposition to orthogonalize Jacobian
                if self.use_qr:
                    Q[i], R[i] = np.linalg.qr(J[i])
            jac_arg = Q if self.use_qr else J
            # Solve transformed robust PCA problem
            L, E, delta_params, numIterInnerEach, Y = self._inner_loop(D.T, jac_arg)
            # Compute new transformations
            for i in xrange(N):
                if self.use_qr:
                    delta_params[i] = np.linalg.lstsq(R[i], delta_params[i])[0]
                    #delta_xi[i] = np.dot(np.linalg.inv(R[i]), delta_xi[i])
                #if np.max(np.abs(delta_params[i] > 2)):
                #    pass
                    #1/0
                transforms[i].params += delta_params[i]
                #XXX: FIX NUMERICAL INACCURACIES IN MAP_COORDINATES, NOT HERE
                transforms[i].params[np.abs(transforms[i].params) < 1e-10]  = 0.0

            maxdp = np.max(np.abs(np.array(delta_params)))
            #print 'maxdp=',maxdp
            # Calculate new objective function
            # Ignore computation when running with fixed L as it is costly.
            if self.L_fixed or not self.compute_objective:
                curObj = np.nan
            else:
                U, s, V = np.linalg.svd(L,full_matrices=False)
                curObj = np.linalg.norm(s,1) + lambda_inner*np.linalg.norm(E.ravel(),1) 
            if self.verbose or self.debug_movies or self.compute_objective:
                print('Scale %d %s, Iter %3d  |  Obj=%3.8f  |  Max param change=%3.8f   ' % (self.current_scale, str(self.scales[self.current_scale]), iterNum, curObj, maxdp))
            if self.debug_movies:
                data_aligned = transform_dataset(data, vol_shape, old_transforms)
                movie_fn = os.path.join(self.debug_directory, 'scale%d_iter%d.mp4' % (self.current_scale, iterNum))
                composite_movie([data,data_aligned,L.T,E.T], vol_shape, movie_fn)
                print('\tMovie at %s' % (ovie_fn))
                #np.savez('scale%d_transforms_iter%d'%iterNum,
                #        transforms=transforms, L=L)
            dtfm = [np.linalg.norm(tfm.params-param_old) for tfm, param_old in
                    zip(transforms, params_old)]
           # print 'MEAN PARAM CHANGE', np.mean(dtfm)
            #objs.append(curObj)
            objs.append(np.mean(dtfm))
            if iterNum >self.max_iter or np.mean(dtfm) < tol:
                #print 'Converged'
                converged = True
            #print objs
            iterNum += 1
        return L.T, E.T, transforms

def center_transforms(transforms, rotation=False):
        mu = np.array([tfm.params[:] for tfm in transforms]).mean(0)
        for transform in transforms:
            transform.params[:3] -= mu[:3]
            if rotation:
                transform.params[3:] -= mu[3:]

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
        #f = np.load('/data/share/facedata.npz')
        f = np.load('/Users/poole/facedata.npz')
        data = f['data']
        vec = data[0]
        vol_shape = f['vol_shape']
        T = 100
        #data, true_transforms, true_inv_transforms = generate_synthetic_dataset(vec, vol_shape, T)
        #data_subsamp = data[::subsamp].copy()
        data_subsamp = data
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
    rasl = RASL(
            rank=40, max_iter=30,sigma=None,lam=1.0,
            normalize=True,
            transform_type='PartialEuclidean',
            border=10,low_rank_method='SVT', debug_movies=True)
    L, E, transforms = rasl.fit(data_subsamp, vol_shape)
    #L, E, transforms = rasl_outer(data_subsamp, vol_shape, outer_rank=None,
    #        inner_rank=40, max_iter=10,sigma=None,lambda_inner=1.0,
    #        lambda_outer=0.1, normalize=True, L_fixed=None,
    #        transform_type='PartialEuclidean',
    #        border=5,low_rank_method='SVT', debug_movies=True)
    center_transforms(transforms)
    Dt = transform_dataset(data_subsamp, vol_shape, transforms)
    composite_movie([data_subsamp,Dt], vol_shape, '/tmp/out_d1.mp4')
    params = np.array([t.params for t in transforms])
    if simulate:
        tparams = np.array([t.params for t in true_inv_transforms])
        print('Batch MAD=',np.mean(np.abs(tparams[::subsamp]-params)))
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
    print('Alignment complete. Took %d seconds.' % (time.time() - tic))
    center_transforms(transforms2)
    offparams = np.array([t.params for t in transforms2])
    if simulate:
        print('Online MAD=',np.mean(np.abs(tparams-offparams)))
    #Dt = transform_dataset(data, vol_shape, transforms)
    Dt2 = transform_dataset(data, vol_shape, transforms2)
    composite_movie([data_subsamp,Dt,Dt2[::subsamp]], vol_shape, '/tmp/out_d2.mp4')
