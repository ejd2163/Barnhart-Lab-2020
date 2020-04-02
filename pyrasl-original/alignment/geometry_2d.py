import numpy as np
from scipy.ndimage.filters import gaussian_filter, sobel
from scipy.ndimage.interpolation import affine_transform, geometric_transform
from rpca import brp, ssgodec


transform_type = 'EUCLIDEAN'

def vector_to_volume(vec, vol_shape):
    return vec.reshape(*vol_shape[::-1]).T

def volume_to_vector(vol):
    return vol.ravel(order='f')

def transform_dataset(data, vol_shape, transforms):
    tdata =  np.zeros_like(data)
    for i in xrange(data.shape[0]):
        tdata[i] = volume_to_vector(imtransform(vector_to_volume(data[i], vol_shape), transforms[i]))
    return tdata

def image_gradients(I, sigma=None):
    if sigma is not None:
        Ismooth = gaussian_filter(I, sigma)
    else:
        Ismooth = I
    grads = []
    for dim in xrange(I.ndim):
        grad = sobel(Ismooth, axis=dim, mode='constant') / 8. 
        grads.append(grad)
    return grads[::-1]

def invert_transform(tfm):
    Rinv = np.linalg.inv(tfm[:,:-1])
    #new_tfm = np.hstack((Rinv, np.dot(Rinv, tfm[:,-1])[:, np.newaxis]))
    new_tfm = np.hstack((Rinv, -np.dot(Rinv, tfm[:,-1])[:, np.newaxis]))
    return new_tfm

def imtransform(I, tfm, order=2, border=10):
    I2 = np.zeros(np.array(I.shape)+1)
    I2[1:,1:] = I
    tfm = tfm.copy()
    Iout = affine_transform(I2, tfm[:, :-1].T, tfm[:,-1][::-1], order=order)
    return Iout[1:,1:]


def parameters_to_transform(params):
    tfm = np.eye(3)[:2]
    if transform_type == 'EUCLIDEAN':
        R = np.array([[np.cos(params[0]), -np.sin(params[0])],
                      [np.sin(params[0]),  np.cos(params[0])]])
        tfm[:2, :2] = R
        tfm[:2, 2] = params[1:]
    elif transform_type == 'SIMILARITY':
        R = np.array([[np.cos(params[1]), -np.sin(params[1])],
                      [np.sin(params[1]),  np.cos(params[1])]])
        tfm[:2, :2] = params[0] * R
        tfm[:2, 2] = params[2:]
    else:
        tfm[0,:] = params[:3]
        tfm[1,:] = params[3:]
    return tfm

def transform_to_parameters(tfm):
    if transform_type == 'EUCLIDEAN':
        x = np.zeros(3)
        theta = np.arccos(tfm[0,0])
        if tfm[1,0] < 0:
            theta = -theta
        x[0] = theta
        x[1:] = tfm[:,-1]
    elif transform_type == 'SIMILARITY':
        x = np.zeros(4)
        sI = np.dot(tfm[:2, :2].T, tfm[:2,:2])
        x1 = np.sqrt(sI[0,0])
        theta = np.arccos(tfm[0,0]/x1)
        if tfm[1,0] < 0:
            theta = -theta
        x2 = theta
        x[0] = x1
        x[1] = x2
        x[2:] = tfm[:,-1]
    else:
        raise NotImplementedError()
    return x

def get_jacobian(Iu, Iv, u, v, tfm):
    params = transform_to_parameters(tfm)
    if transform_type == 'EUCLIDEAN':
        x1 = params[0]
        J = [Iu * (-np.sin(x1)*u - np.cos(x1)*v ) + Iv * (np.cos(x1)*u - np.sin(x1) * v ),
             Iu, Iv]
    elif transform_type == 'SIMILARITY':
        x1 = params[0]
        x2 = params[1]
        J = [Iu * (np.cos(x2) * u - np.sin(x2)*v) + Iv * (np.sin(x2) * u + np.cos(x2)*v), 
             Iu * (-x1*np.sin(x2)*u-x1*np.cos(x2)*v) + Iv * (x1*np.cos(x2)*u - x1*np.sin(x2)*v), Iu, Iv]
    else:
        J = [Iu * u, Iu * v, Iu, Iv*u, Iv*v, Iv]
    return J

def image_jacobian(I, tfm, sigma=8, normalize=False): 
    sI = np.dot(tfm[:2, :2].T, tfm[:2,:2])
    x1 = np.sqrt(sI[0,0])
    theta = np.arccos(tfm[0,0]/x1)
    if tfm[1,0] < 0:
        theta = -theta
    x2 = theta
    #XXX: take in transformed image?
    grads = image_gradients(I, sigma)
    # Transform image
    It = imtransform(I, tfm)
    grads = [imtransform(grad, tfm) for grad in grads]
    #grads = image_gradients(It, sigma)
    Iu = grads[0] 
    Iv = grads[1]
    if normalize:
        normI = np.linalg.norm(It.ravel())
    else:
        normI = 1.
    # Normalize Iu, Iv, I
    Iu = (1. / normI) * Iu - (Iu * It).sum()/ (normI**3) * It
    Iv = (1. / normI) * Iv - (Iv * It).sum()/ (normI**3) * It
    It /= normI
    # Compute Jacobian
    v = np.tile(np.arange(I.shape[0]).reshape(-1, 1)+1, [1, I.shape[1]])
    u = np.tile(np.arange(I.shape[1]).reshape(1,-1)  + 1, [I.shape[0], 1])
    J = get_jacobian(Iu, Iv, u, v, tfm)
    # Vectorize elements
    J = np.vstack([volume_to_vector(j) for j in J]).T
    return volume_to_vector(It), J
