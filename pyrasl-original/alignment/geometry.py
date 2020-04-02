#
# Geometric operations and primitives used by RASL.
#

import numpy as np
from scipy.ndimage.filters import gaussian_filter, sobel
from scipy.ndimage import interpolation
from util.volume import vector_to_volume, volume_to_vector, zero_border
from scipy.ndimage import map_coordinates

class Transform(object):
    """Abstract class for geometric transforms."""
    def jacobian(self): raise NotImplementedError()
    def apply(self, im): raise NotImplementedError()
    def inverse(self): raise NotImplementedError()
    def resize(self, factor):
        self.params[:3] *= factor

class TranslationTransform(Transform):
    def __init__(self, shift=None):
        """Translation in 3d.

        Args: 
            shift: 3d vector of translations (x, y, z)
        """
        if shift is None:
            shift = np.zeros(3)
        assert len(shift) == 3
        self.params = shift

    def jacobian(self, Ix, Iy, Iz, x=None, y=None, z=None):
        J = [Ix, Iy, Iz]
        return J


    def apply(self, im):
        return imtransform(im, shift=self.params)
        
    def inverse(self):
        return TranslationTransform(-self.params)

    def matrix(self):
        return build_transform(self.params)

class EuclideanTransform(Transform):
    def __init__(self, shift=None, rotation=None):
        """Translation and rotation in 3d.

        Args: 
            shift: 3d vector of translations (x, y, z)
            rotation: 3d vector of rotations (xy angle, xz angle, yz angle)
        """
        if shift is None:
            shift = np.zeros(3)
        if rotation is None:
            rotation = np.zeros(3)
        assert len(shift) == 3
        assert len(rotation) == 3
        self.params = np.hstack((shift, rotation))

    def jacobian(self, Ix, Iy, Iz, x, y, z):
        stheta = np.sin(self.params[3])
        ctheta = np.cos(self.params[3])
        sphi = np.sin(self.params[4])
        cphi = np.cos(self.params[4])
        spsi = np.sin(self.params[5])
        cpsi = np.cos(self.params[5])
        #TODO(ben): Verify correctness of Jacobian
        Jr1 = Ix * (-x * cphi * stheta + z * (-cpsi * stheta * sphi + ctheta * spsi) + y * (-ctheta * cpsi - stheta * sphi * spsi)) 
        Jr1 +=  Iy * (x * ctheta * cphi + z * (ctheta*cpsi*sphi + stheta*spsi) + y*(-cpsi*stheta+ctheta*sphi*spsi))
        Jr2 = Ix * (z * ctheta * cphi * cpsi - x * ctheta * sphi + y * ctheta * cphi * spsi)
        Jr2 += Iy * (z * cphi * cpsi * stheta - x * stheta * sphi + y * cphi * stheta * spsi)
        Jr2 += Iz * (-x * cphi - z * cpsi * sphi - y * sphi * spsi)
        Jr3 = Ix * (y * (ctheta * cpsi * sphi + stheta * spsi) + z * (cpsi *
            stheta - ctheta * sphi * spsi) )
        Jr3 += Iy * (y * (cpsi * stheta * sphi - ctheta * spsi) + z * (-ctheta
            * cpsi - stheta * sphi * spsi))
        Jr3 += Iz * (y * cphi * cpsi - z * cphi * spsi)
        J = [Ix, Iy, Iz, Jr1, Jr2, Jr3]
        return J

    def apply(self, im):
        return imtransform(im, shift=self.params[:3], rotation=self.params[3:])

    def matrix(self):
        return build_transform(self.params[:3],  self.params[3:])

    def inverse(self):
        #TODO(ben): implement this
        raise NotImplementedError()

class PartialEuclideanTransform(Transform):
    def __init__(self, shift=None, rotation=0.0):
        """Translation in 3d, and rotation in xy plane.
        
        Args: 
            shift: 3d vector of translations (x, y, z)
            rotation: scalar for xy rotation angle in radians
        """
        if shift is None:
            shift = np.zeros(3)
        self.params = np.hstack((shift, rotation))

    def jacobian(self, Ix, Iy, Iz, x, y, z):
        stheta = np.sin(self.params[3])
        ctheta = np.cos(self.params[3])
        Jr1 = Ix * (-x * stheta - y * ctheta ) 
        Jr1 +=  Iy * (x * ctheta - y * -stheta)
        J = [Ix, Iy, Iz, Jr1]
        return J

    def apply(self, im):
        return imtransform(im, shift=self.params[:3], rotation=np.array([self.params[3], 0.0, 0.0]))

    def matrix(self):
        rot = np.zeros(3)
        rot[0] = self.params[3]
        return build_transform(self.params[:3],  rot)

    def inverse(self):
        return PartialEuclideanTransform(shift=-self.params[:3], rotation=-self.params[3])



class AffineTransform(Transform):
    def __init__(self, shift=None, rotation=None):
        """Translation in 3d, and rotation in xy plane.
        
        Args: 
            shift: 3d vector of translations (x, y, z)
            rotation: scalar for xy rotation angle in radians
        """
        if shift is None:
            shift = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)
        self.params = np.hstack((shift, rotation.ravel()))

    def jacobian(self, Ix, Iy, Iz, x, y, z):
        J = [Ix, Iy, Iz]
        from itertools import product
        js = product((Ix,Iy,Iz), (x,y,z))
        for grad, point in js:
            J.append(grad*point)
        return J

  #  def apply(self, im):
  #      return imtransform(im, shift=self.params[:3], rotation=np.array([self.params[3], 0.0, 0.0]))

    def matrix(self):
        A = np.eye(4)
        A[:3, 3] = self.params[:3]
        A[:3,:3] = self.params[3:].reshape(3,3)
        return A

class HomographyTransform(Transform):
    def __init__(self, shift=None):
        if shift is None:
            shift = np.zeros(3)
        self.params = np.hstack((shift, np.eye(2).ravel(), [0,0]))

    def jacobian(self, Ix, Iy, Iz, x, y, z):
        A = self.matrix()
        X = A[0,0] * x + A[0,1] * y + A[0,3]
        Y = A[1,0] * x + A[1,1] * y + A[1, 3]
        N = A[3,0] * x + A[3,1] * y + 1

        J = [Ix / N, Iy / N, np.zeros_like(Iz), Ix * x / N, Ix * y / N, Iy * x / N, Iy * y / N, 
            (-Ix * X * x / (N**2) - Iy * Y * x / (N**2)),
            (-Ix * X * y / (N**2) - Iy * Y * y / (N**2))]
        return J

    def matrix(self):
        A = np.eye(4)
        A[:3, 3] = self.params[:3]
        A[[0,1,3],:2] =  self.params[3:].reshape(3,2)
        return A


# Global maping from transform names to transform classes
TRANSFORMS = {
    'Translation': TranslationTransform,
    'Euclidean': EuclideanTransform,
    'PartialEuclidean': PartialEuclideanTransform,
    'Affine':AffineTransform,
    'Homography': HomographyTransform }


def transform_vol(vol, transform, grid=None, order=3):
    if grid is None:
        grid = GridTransformer(vol.shape)
    A = transform.matrix()
    return grid.transform_vol(vol, A, order=order)

def transform_frame(frame, vol_shape, transform, grid=None, order=3):
    """Transform  a single volume

    Args:
        frame: vector of size nvoxels
        vol_shape: 
        transform: Transform object
    """
    return volume_to_vector(transform_vol(vector_to_volume(frame, vol_shape), transform, grid, order=order))

def transform_dataset(data, vol_shape, transforms, grid=None, order=3):
    """Transform each volume in a dataset.

    Args:
        data: nframes x nvoxels
        vol_shape: 
        transforms: list of Transform objects
    """
    tdata =  np.zeros_like(data)
    if grid is None:
        grid = GridTransformer(vol_shape[::-1])
    for i in xrange(data.shape[0]):
        tdata[i] = transform_frame(data[i], vol_shape, transforms[i], grid, order=order)
    return tdata

def image_gradients(I, sigma=None):
    """Compute gradient of image in x,y,z directions."""
    if sigma is not None:
        Ismooth = gaussian_filter(I, sigma)
    else:
        Ismooth = I
    grads = []
    for dim in xrange(I.ndim):
        grad = sobel(Ismooth, axis=dim, mode='constant') / 8. 
        grads.append(grad)
    return grads

def imtransform(I, shift=None, rotation=None, order=1, rorder=3):
    """Translate and rotate a 3d volume.

    Args:
        I: 3d volume
        shift: 3d vector of translations (x, y, z)
        rotation: 3d vector of rotations (xy angle, xz angle, yz angle)
        order: Spline order for interpolation.
    Returns:
        Transformed volume.
    """
    if rotation is not None:
        if rotation[2] != 0.0:
            I = interpolation.rotate(I, -np.rad2deg(rotation[2]), axes=(1,2), reshape=False, order=rorder)
        if rotation[1] != 0.0:
            I = interpolation.rotate(I, -np.rad2deg(rotation[1]), axes=(0,2), reshape=False, order=rorder)
        if rotation[0] != 0.0:
            I = interpolation.rotate(I, -np.rad2deg(rotation[0]), axes=(0,1), reshape=False, order=rorder)
    if shift is not None:
        I = interpolation.shift(I, -shift, order=order)
    return I


def image_jacobian(I, tfm, grid, sigma=None, normalize=False, border=1, order=1):
    """Compute Jacobian of volume w.r.t. transformation parameters

    Args:
        I: volume
        tfm: Transform object
        sigma: smoothing bandwidth for gradients (None for no smoothing)
        normalize: Whether to normalize images before aligning.
        border: Number or tuple of border sizes to zero after transforming.
    Returns:
        It: vectorized version of transformed volume
        J: nvoxels x nparameters Jacobian matrix
    """
    # Compute gradients
    grads = image_gradients(I, sigma)
    # Transform image and gradients
    # Note: does not work if we compute gradients after transforming (borders?)
    #It = zero_border(tfm.apply(I), border)
    It = transform_vol(I, tfm, grid, order=order)
    grads = [transform_vol(grad, tfm, grid, order=order) for grad in grads]
    Iu = grads[0]
    Iv = grads[1]
    Iw = grads[2]
    # Normalize
    if normalize:
        normI = np.linalg.norm(It.ravel())
    else:
        normI = 1.
    Iu = (1. / normI) * Iu - (Iu * It).sum()/ (normI**3) * It
    Iv = (1. / normI) * Iv - (Iv * It).sum()/ (normI**3) * It
    Iw = (1. / normI) * Iw - (Iw * It).sum()/ (normI**3) * It
    It /= normI
    # Static matrices needed by jacobian function.
    # We generate them once and store as a static variable in this function.
    # Compute Jacobian
    J = tfm.jacobian(Iu, Iv, Iw, grid.homo_points[0,...], grid.homo_points[1,...], grid.homo_points[2,...])
    # Vectorize and return elements
    J = np.column_stack([volume_to_vector(zero_border(j,border)) for j in J])
    It = zero_border(It, border)
    return volume_to_vector(It), J


def build_transform(shift=None, rot=None):
    if shift is None:
        shift = np.zeros(3)
    if rot is None:
        rot = np.zeros(3)
    c = np.cos(rot)
    s = np.sin(rot)
    tran = [[1,     0,     0,    shift[0]],
            [0,     1,     0,    shift[1]], 
            [0,     0,     1,    shift[2]],
            [0,     0,     0,    1]]
    xrot = [[c[0], -s[0],  0,    0], 
            [s[0],  c[0],  0,    0],
            [0,     0,     1,    0],
            [0,     0,     0,    1]]
    yrot = [[c[1],  0,     s[1], 0,], 
            [0,     1,     0,    0],
            [-s[1], 0,     c[1], 0],
            [0,     0,     0,    1]]
    zrot = [[1,     0,     0,    0],
            [0,     c[2], -s[2], 0], 
            [0,     s[2],  c[2], 0],
            [0,     0,     0,    1]]
    A = np.dot(tran, np.dot(xrot, np.dot(yrot, zrot)))
    return A 

class GridTransformer(object):
    def __init__(self, dims, center=None):
        """Class to represent and transform a fixed grid of points over indices into a volume.

        Args:
            dims: vol_shape[::-1]
            center: center location of grid, defaults to the centroid.
                    Affine transfroms will be applied to the center-subtracted grid,
                    so the center corresponds to the point of rotation.
        """
        self.dims = np.array(dims)
        if center is None:
            center = (self.dims -1) / 2.0
        ix, iy, iz =  np.mgrid[:dims[0],:dims[1],:dims[2]]

        self.center = center
        self.homo_points = np.ones(np.hstack(((4,) ,dims))  )
        self.homo_points[:-1, ...] = np.array((ix,iy,iz))
        self.raw_points = self.homo_points.copy()
        self.world_to_index_tfm = build_transform(self.center, (0,0,0))
        self.index_to_world_tfm = build_transform(-self.center, (0,0,0))
        self.homo_points = self.transform_grid_world(self.index_to_world_tfm)

    def transform_grid_world(self, A):
        """Get the grid of points in world space after applying the given affine transform.
        """
        return np.tensordot(A.T, self.homo_points, axes=(0,0))

    def transform_grid(self, A):
        """Get the grid of points in index space after applying the given affine transform
           in world space.

        Args:
            A: 4x4 Affine transformation matrix
        Returns:
            y: 4 x dims[0] x dims[1] x dims[2] matrix containing the grid
        """
        return self.transform_grid_world(np.dot(self.world_to_index_tfm, A))

    def transform_vol(self, vol, A, **kwargs):
        new_grid = self.transform_grid(A)[:3]
        transformed_vol = map_coordinates(vol, new_grid, cval=0.0, **kwargs)
        return transformed_vol

