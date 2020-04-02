#
# Assorted matrix approximation techniques.
#

import numpy as np

def soft_threshold(X, lam):
    """Elementwise soft-thresholding of vector or matrix."""
    return np.sign(X) * np.maximum(0.0, np.abs(X) - lam)

def singular_value_soft_threshold(X, lam):
    """Low-rank matrix approximation using singular value soft-thresolding."""
    U, s, V = np.linalg.svd(X, full_matrices=False)
    s = soft_threshold(s, lam)
    idx = np.nonzero(s)[0]
    #print len(idx)
    U = U[:, idx]
    V = np.dot(np.diag(s[idx]), V[idx, :]).T
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
    V, R = np.linalg.qr(Y2, mode='full')
    U = np.dot(X,V)
    if transpose:
        U, V = V, U
    return U, V

def brpnn(X, rank, lam, power=6):
    if X.shape[0] > X.shape[1]:
        transpose = True
    else:
        transpose = False
    if transpose:
        X = X.T
    U, V = brp(X, rank, power)
    Q,R = np.linalg.qr(V)
    Unew, Vnew = singular_value_soft_threshold(np.dot(U, R.T), lam)
    U2 = U[:, :Vnew.shape[1]]
    V2 = np.dot(Q, Vnew)
    #U, V = U2, V2
    if transpose:
        U, V = V, U
    return U, V

low_rank_methods = {
            'SVD': truncated_svd,
            'BRP': brp,
            'NN': singular_value_soft_threshold,
            'BRPNN': brpnn }

def low_rank_approximation(X, method='BRPNN', **kwargs):
    """Compute low-rank matrix approximation.

    Args:
        X: data matrix
        method: must be one of
            SVD: truncated SVD, must specify rank
            BRP: bilateral random projections, specify rank
            NN: nuclear norm, specify lam
            BRPNN: BRP followed by NN, specify rank and lam
        **kwargs: keyword args to be passed to method
    Returns:
        Matrices U, V such that X is approximately UV^T
    """
    if method in ['SVD', 'BRP', 'BRPNN'] and 'rank' not in kwargs:
        raise ValueError('Must specify rank when using %s for low-rank approximation.'%method)
    if method in ['NN', 'BRPNN'] and 'lam' not in kwargs:
        raise ValueError('Must specify lam when using %s for low-rank approximation.'%method)
    if method not in low_rank_methods:
        raise ValueError('Unknown method %s.' % method)
    return low_rank_methods[method](X, **kwargs)


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


