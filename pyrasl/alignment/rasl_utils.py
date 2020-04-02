import numpy as np
from scipy.ndimage.filters import gaussian_filter, sobel
from scipy.ndimage.interpolation import affine_transform

def image_gradients(I, sigma=None):
    if sigma is not None:
        Ismooth = gaussian_filter(I, sigma)
    else:
        Ismooth = I
    grads = []
    for dim in xrange(I.ndim):
        grad = sobel(Ismooth, axis=dim, mode='constant') / 8. 
        grads.append(grad)
    return grads

def imtransform(I, tfm, order=2):
    return affine_transform(I, tfm[:, :-1], tfm[:,-1], order=order)

def image_jacobian(I, sigma=8): 
    #XXX: take in transformed image?
    grads = image_gradients(I, sigma)
    # Transform image
    It = I
    #It = imtransform(I, tfm)
    #grads = [imtransform(grad, tfm) for grad in grads]
    Iu = grads[0] 
    Iv = grads[1]
    normI = np.linalg.norm(It.ravel())
    # Normalize Iu, Iv, I
    Iu = (1. / normI) * Iu - (Iu * It).sum()/ (normI**3) * It
    Iv = (1. / normI) * Iv - (Iv * It).sum()/ (normI**3) * It
    It /= normI
    # Compute Jacobian
    u = np.arange(I.shape[1]).reshape(1,-1) + 1
    v = np.arange(I.shape[0]).reshape(-1, 1) + 1
    J = [Iu * u, Iv * v, Iu, Iv*u, Iv*v, Iv]
    # Vectorize elements
    J = np.vstack([j.ravel() for j in J]).T
    return J

    




