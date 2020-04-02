  #
# Assorted matrix approximation techniques.
#

import numpy as np

def soft_threshold(X, lam):
    """Elementwise soft-thresholding of vector or matrix."""
    return np.sign(X) * np.maximum(0.0, np.abs(X) - lam)


def singular_value_threshold_full(X, lam):
    """Low-rank matrix approximation using singular value thresolding."""
    U, s, V = np.linalg.svd(X, full_matrices=False)
    s = soft_threshold(s, lam)
    idx = np.nonzero(s)[0]
    #print len(idx)
    U = U[:, idx]
    V = V[idx, :].T
    return U, s[idx], V

def singular_value_threshold(X, lam):
    """Low-rank matrix approximation using singular value thresolding."""
    U, s, V = singular_value_threshold_full(X, lam)
    U = np.dot(U, np.diag(s))
    return U, V

def truncated_svd(X, rank):
    """Low-rank matrix approximation using truncated SVD."""
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U = U[:, :rank] 
    V = np.dot(np.diag(s[:rank]), V[:rank, :]).T
    return U, V

def brp(X, rank, power=6):
    """Low-rank matrix approximating using bilateral random projections

    Args:
        X: data matrix
        rank: desired rank
        power: iterations of power method
    """
    transpose = X.shape[0] < X.shape[1]
    if transpose:
        X = X.T
    Y2 = np.random.randn(X.shape[1], rank)
    for i in xrange(power):
        Y1 = np.dot(X, Y2)
        Y2 = np.dot(X.T, Y1)
    V, R = np.linalg.qr(Y2)
    U = np.dot(X,V)
    if transpose:
        U, V = V, U
    return U, V

def brpsvt(X, rank, lam, power=6):
    """Low-rank matrix approximating using BRP and singular value thresholding.

    Computes an initial low-rank approximation with BRP, then reduces this
    initial approximation using SVT. This saves the computational overhead
    of computing the full SVD when doing normal SVT.

    Args:
        X: data matrix
        rank: desired rank
        power: iterations of power method
    """
    if X.shape[0] > X.shape[1]:
        transpose = True
    else:
        transpose = False
    if transpose:
        X = X.T
    U, V = brp(X, rank, power)
    Q,R = np.linalg.qr(V)
    Unew, Vnew = singular_value_threshold(np.dot(U, R.T), lam)
    U = Unew
    V = np.dot(Q, Vnew)
    if transpose:
        U, V = V, U
    return U, V

low_rank_methods = {
            'SVD': truncated_svd,
            'BRP': brp,
            'SVT': singular_value_threshold,
            'BRPSVT': brpsvt }

def low_rank_approximation(X, method='BRPSVT', **kwargs):
    """Compute low-rank matrix approximation.

    Args:
        X: data matrix
        method: must be one of
            SVD: truncated SVD, must specify rank
            BRP: bilateral random projections, specify rank
            SVT: singular value thresholding, specify lam
            BRPSVT: BRP followed by SVT, specify rank and lam
        **kwargs: keyword args to be passed to method
    Returns:
        Matrices U, V such that X is approximately UV^T
    """
    if method in ['SVD', 'BRP', 'BRPSVT'] and 'rank' not in kwargs:
        raise ValueError('Must specify rank when using %s for low-rank approximation.'%method)
    if method in ['SVT', 'BRPSVT'] and 'lam' not in kwargs:
        raise ValueError('Must specify lam when using %s for low-rank approximation.'%method)
    if method not in low_rank_methods:
        raise ValueError('Unknown method %s.' % method)
    return low_rank_methods[method](X, **kwargs)


def splr(X, tau, gamma, theta=0.1):
    """Sparse and low-rank matrix approximation. 

    Args:
        X: data matrix
        tau: nuclear norm penalty
        gamma: L1 penalty
        theta: learning rate
    Returns:
        S: sparse and low-rank matrix the size of X
    """
    S = X.copy()
    normX = np.linalg.norm(X)
    for i in xrange(10):
        Sprev = S.copy()
        grad = 2 * (S - X)
        S -= theta * grad
        # prox for nuclear norm
        Snorm1 = np.linalg.norm(S)
        U, V = singular_value_threshold(S, theta * tau)
        print 'rank=',U.shape[1]
        S = np.dot(U, V.T)
        Snorm2 = np.linalg.norm(S)
        # prox for l1
        S = soft_threshold(S, theta * gamma)
        # No projection needed
        Snorm3 = np.linalg.norm(S)
        print i, np.linalg.norm(S-Sprev)
        print 'normrat=',Snorm1/normX, Snorm2/normX, Snorm3/normX
    return S


def ssgodec(X, rank, tau, power=6, max_iter=10, error_bound=1e-3):
    """Low-rank plus sparse decomposition using SSGoDec.

    Decomposes X into a low-rank and sparse component using the
    semi-soft go decomposition from: 
    https://sites.google.com/site/godecomposition/code

    Args:
        X: matrix
        rank: desired rank of L
        tau: sparsity penalty
        power: iterations of power method to run
        max_iter: maximum number of iterations
        error_bound: lower bound for relative error
    Returns:
        L: low-rank matrix
        S: sparse matrix
        rmse: list of rmses per iteration
        error: final rmse
        U: left factor of L
        V: right factor of L
    """
    m, n = X.shape
    if m < n:
        X = X.T
    # Initialization
    L = X
    S = np.zeros_like(X)#sps.lil_matrix(X.shape)
    norm = lambda x: np.linalg.norm(x.flatten())
    normX = norm(X)
    rmse = []
    iter = 1
    while True:
        # Compute low rank component
        U, V = brp(L, rank, power)
        L_new = np.dot(U, V.T)
        # Update Residual
        T = L - L_new + S
        # Compute sparse component
        S = soft_thresh(T, tau)
        # Gaussian component (X-L-S)
        T = T - S
        # Overwrite old L (should be able to reduce memory footprint)
        L = L_new
        rmse.append(norm(T)/normX)
        if iter > 1:
            relerror = (rmse[-2]-rmse[-1])/rmse[-1]
        else:
            relerror = np.inf
        if relerror < error_bound or iter > max_iter:
            print 'breaking with rmse=', rmse[-1]
            break
        else:
            L = L + T
        print 'iter %d, nnz=%g, err=%g' % (iter,len(np.nonzero(S)[0])/float(np.prod(S.shape)),rmse[-1])
        iter += 1
    LS = L + S
    error = rmse[-1]
    # Transpose outputs if required
    if m < n:
        #XXX: .copy() these so array is in right layout?
        LS = LS.T
        L = L.T 
        S = S.T
        U, V = V, U.T
    return  L, S, rmse, error, U, V

def flatten(X, d):
    dims = np.roll(np.arange(X.ndim), -d)
    X = np.transpose(X, dims)
    return np.reshape(X, (X.shape[0], -1))

def flatten_adj(X, sz, d):
    dims_fwd = np.roll(np.arange(len(sz)), -d)
    dims = np.roll(np.arange(len(sz)), d)
    return np.transpose(X.reshape(sz[dims_fwd]), dims)


def tensor_admm(data, lam, gammas, eta=1.0, tol=1e-4, maxiter=10, verbose=False):
    eta /= data.std() 
    #XXX: destroys data
    X = np.zeros_like(data)#X.copy()
    nd = X.ndim
    sz = np.array(X.shape)
    n = np.prod(sz)
    Z = [np.zeros((sz[i], n/sz[i])) for i in xrange(X.ndim)]
    Y = [np.zeros((sz[i], n/sz[i])) for i in xrange(X.ndim)]
    S = [None for i in xrange(X.ndim)]

    dnz = data != 0
    viol = np.zeros(nd)

    fval = np.zeros(maxiter)
    gval = np.zeros(maxiter)
    for kk in xrange(maxiter):
        # UPDATE X (low rank approx of full dataset)
        X1 = np.zeros_like(X)
        for jj in xrange(X.ndim):
            X1 -= flatten_adj(Y[jj] - eta * Z[jj], sz, jj)
        if lam > 0.0:
            X1 += data / lam
            X = X1 / ( dnz / lam + nd * eta);
        else:
            X = X1 / (nd * deta)
            X = data
            #XXX: deal with partial observations
        # UPDATE Z
        for jj in xrange(X.ndim):
#            Z[jj] = singular_value_threshold(Y[jj] / eta + flatten(X, jj) - Z[jj], gammas[jj]/eta)
            UU, S[jj], VV = singular_value_threshold_full(Y[jj] / eta + flatten(X, jj), gammas[jj]/eta)
            Z[jj] = np.dot(UU, np.dot(np.diag(S[jj]), VV.T))
        for jj in xrange(X.ndim):
            V = flatten(X, jj) - Z[jj]
            Y[jj] += eta * V
            viol[jj] = np.linalg.norm(V.ravel())

        # Compute objective
        G = np.zeros_like(X)
        for jj in xrange(nd):
            fval[kk] += np.linalg.svd(flatten(X,jj), compute_uv=False).sum()
            G += flatten_adj(Y[jj], sz, jj)
        if lam > 0.0:
            fval[kk] += 0.5 * np.sum((X-data)**2)/lam;
            G += (X - data)/lam
        else:
            # G(ind)=0
            pass
        gval[kk] = np.linalg.norm(G.ravel())

        if kk > 1 and np.max(viol) < tol and gval[kk] < tol:
            break

        if verbose:
            print '%d: f=%g, g=%g' % (kk, fval[kk], gval[kk])
            for jj in xrange(X.ndim):
                print '\t%d - n=%d, sum=%g' % (jj, len(S[jj]), S[jj].sum())
    return X








