# __BEGIN_LICENSE__
#
# Copyright 2012 Stanford University.  All rights reserved.
#
# __END_LICENSE__

# Major Imports 
import os, sys, time, multiprocessing, h5py
import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter, laplace

#======================================================================
#                     MAIN CALLABLE FUNCTION
#======================================================================

def interest_point_alignment(input_filename, output_filename, options):

    if options.save_key_frames or options.tune:
        output_keydir = os.path.join(os.path.dirname(output_filename), 'alignment_keyframes')
        if not os.path.exists(output_keydir):
            os.makedirs(output_keydir)

    # COMPUTE AN AVG "ANCHOR" IMAGE
    #
    # Everything gets aligned to this image... it serves as the basis
    # for aligning images.  To form it, we average together the first
    # 50 frames of video.
    #
    # IMPORTANT NOTE: This needs to be done in a separate thread,
    # because h5py seems to get really flakey when you open a file in
    # the main thread, and then close and re-open it in child threads.
    # The Internet reports that child sub-processes inherit the h5py
    # state information, so there is something funny going on here.
    # For now the workaround is to avoid opening the input file in the
    # main thread.
    tic = time.time()
    print '\t--> Computing reference frame'
    from multiprocessing import Pool
    pool = Pool(processes=options.num_threads)
    imap_data = [ (input_filename, options.keyframe_spacing, options.keyframe_offset, 1, options.mask) ]
    results = pool.imap(compute_key_frames_imap, imap_data)
    for r in results:
        t_start = 0
        t_end = r[0]
        reference_vols = r[1]
        vol_shape = r[2]
        reference_t = r[3]
    pool.close()
    pool.join()

    # Compute interest points for the reference volumes
    reference_pts = {}
    for v in reference_vols.keys():
        reference_pts[v] = ip_detect(reference_vols[v], options.sigma, options.neighborhood, options.threshold)

    # Save key frames if requested
    if options.save_key_frames or options.tune:
        print '\t--> Saving diagnostic key frames to %s'%output_keydir
        for k in reference_vols.keys():
            vol = reference_vols[k].copy()
            vol = np.sqrt(vol)
            vol = (vol - vol.min()) / (vol.max() - vol.min())
            import cv2
            imgxy = np.tile(((vol.max(2)-vol.min())/(vol.max()-vol.min()))[:,:,np.newaxis],(1,1,3))
            imgxz = np.tile(((vol.max(0)-vol.min())/(vol.max()-vol.min()))[:,:,np.newaxis],(1,1,3))
            imgyz = np.tile(((vol.max(1)-vol.min())/(vol.max()-vol.min()))[:,:,np.newaxis],(1,1,3))        
            for p in range(reference_pts[k].shape[0]):
                cv2.circle(imgxy,(reference_pts[k][p,1],reference_pts[k][p,0]),2,(1,0,0),-1)
                cv2.circle(imgxz,(reference_pts[k][p,2],reference_pts[k][p,1]),2,(1,0,0),-1)
                cv2.circle(imgyz,(reference_pts[k][p,2],reference_pts[k][p,0]),2,(1,0,0),-1)
            imgortho = np.zeros((imgxy.shape[0] + imgyz.shape[1],
                                 imgxy.shape[1] + imgxz.shape[1],
                                 imgxy.shape[2]), dtype=imgxy.dtype)
            imgortho[0:imgxy.shape[0],0:imgxy.shape[1],:] = imgxy
            imgortho[imgxy.shape[0]:,0:imgxy.shape[1],:] = np.flipud(imgxz.transpose(1,0,2))
            imgortho[0:imgxy.shape[0],imgxy.shape[1]:,:] = np.fliplr(imgyz)
            cv2.imwrite(os.path.join(output_keydir,'key%d.png'%k), imgortho*255) #scaling hack

    if options.tune:
        sys.exit(0)

    # Match interest points for reference volumes
    reference_best_matches = {}
    reference_translations = {}
    reference_rotations = {}
    reference_scales = {}
    for v in range(len(reference_vols.keys())-1):
        reference_best_matches[v] = match_points(reference_pts[v],
                                                 reference_pts[v+1], 10) # Rejection threshold = 10 px
        if options.transonly:
            (reference_scales[v], reference_rotations[v], reference_translations[v]) = compute_transform(reference_pts[v],
                                                                                                         reference_pts[v+1],
                                                                                                         reference_best_matches[v])
        else:
            (reference_scales[v], reference_rotations[v], reference_translations[v]) = compute_geometric_transform(reference_pts[v],
                                                                                                                   reference_pts[v+1],
                                                                                                                   reference_best_matches[v])
        num_bad_matches = sum([x == None for x in reference_best_matches[v]])
        num_good_matches = reference_pts[v].shape[0]-num_bad_matches
        s = '\t    [ Key frame %d to %d ] : matched %d / %d interest points' % (v+1, v+2,
                                                                                      num_good_matches,
                                                                                      reference_pts[v].shape[0])
        print s, '    Translation: ', reference_translations[v]
        print '\tRotation: ', reference_rotations[v]
        print '\tScale: ', reference_scales[v]

    # Compute the minimum pixel value in the reference volumes.  This
    # becomes the contant value we use for edge extension as we shift
    # things around.
    minval = 1e9
    for k in reference_vols.keys():
        minval = min(reference_vols[k].min(), minval)
    print '\t--> Constant edge extension set to minimum keyframe voxel value: ', minval

    print "\t    Reference frame generation took %0.2f seconds." % (time.time() - tic)
    
    # Initial time range, potentially modified by the user below.
    if options.time_range != None:
        from util.cmd_line_options import parse_time_range_string
        (t_start, t_end) = parse_time_range_string(options.time_range)
        print '\t--> Using user-defined time range: [', t_start, t_end,']'

    # Open the output file, prepare to write results to disk.
    output_file = h5py.File(output_filename,'w')
    output_file.create_dataset('vol_shape', data=vol_shape)
    output_volume = output_file.create_dataset('timeseries_volume', (t_end-t_start, np.prod(vol_shape)), 'f')
   
    keyframe_vec = reference_vols.keys()
    output_file.create_dataset('keyframes', data=keyframe_vec)

    # ALIGN IMAGES
    #
    # Run across multiple threads
    print '\t--> Aligning images.'
    tic = time.time()
    pool = Pool(processes=options.num_threads)
    imap_data = [ (input_filename, i, options.sigma, reference_pts, reference_t, reference_scales, reference_rotations, reference_translations,
                   minval, options.mask, options.neighborhood, options.threshold, options.transonly)  for i in xrange(t_start, t_end) ]
    results = pool.imap(align_image_imap, imap_data, chunksize = 1) #chunksize = min(len(imap_data)/(options.num_threads*2),20)
    interest_points = {}
    best_matches = {}
    transforms = {}
    aligned_vols = {}
    reference_ndx = {}
    for r in results:
        time_slice = r[0]
        interest_points[time_slice] = r[1]
        best_matches[time_slice] = r[2]
        transforms[time_slice] = r[3]
        
        aligned_vols[time_slice] = r[4]
        reference_ndx[time_slice] = r[5]

        output_volume[time_slice-t_start, :] = np.reshape(aligned_vols[time_slice],
                                                          np.prod(vol_shape), order='f')
        
        num_bad_matches = sum([x == None for x in best_matches[time_slice]])
        num_good_matches = interest_points[time_slice].shape[0]-num_bad_matches
        s = "\t    [ Slice %d ]: Matched %d / %d interest points to reference volume %d " % (time_slice,
                                                                       num_good_matches,
                                                                       interest_points[time_slice].shape[0],
                                                                       reference_ndx[time_slice])
        print s, '     Translation:', transforms[time_slice][2]
        print '\t\tRotation:',transforms[time_slice][1]
        print '\t\tScale:',transforms[time_slice][0]
    pool.close()
    pool.join()

    print "\t    Image alignment took %0.2f seconds on %d threads." % (time.time() - tic, options.num_threads)

    output_file.close()

    return True

#----------------------------------------------------------------------------#
#                           UTILITY FUNCTIONS
#----------------------------------------------------------------------------#

class GeometricTransform(object):
    """
    Solves for geometric transform that minimizes least sq distance between to a
    corresponding set of data points.
    """

    def __init__(self,dim=3,bScale=False):
        self.rot = None
        self.trans = None
        self.scale = None
        self.bScale = bScale
        self.dim = dim

    def fit(self,data):      
        """
        Data must be #points x (nDim x 2), first ndim columns are reference points;
        second ndim columns are data points. 
        Returns a model (scale, orthonormrot, trans) that transforms to data points to match the ref points.
        """
        A = data[:,0:self.dim]
        B = data[:,self.dim:]

        #Procrustean: CODE TO SOLVE FOR OPTIMAL (by least squares) EUCLIDEAN TRANSFORM (currently allows reflections):
        bReflection = True

        muA = np.mean(A,axis=0)
        muB = np.mean(B,axis=0)
        A0 = A - np.tile(muA,[A.shape[0],1])
        B0 = B - np.tile(muB,[B.shape[0],1])

        ssqA = np.sum(A0**2)
        ssqB = np.sum(B0**2)
        normA = np.sqrt(ssqA)
        normB = np.sqrt(ssqB)

        A0 = A0 / normA
        B0 = B0 / normB

        X = np.dot(A0.T,B0)
        [L, D, M] = np.linalg.svd(X)
        rotation = np.dot(M.T,L.T)

        traceTA = sum(D)
        if self.bScale:
            scale = traceTA * normA / normB
        else:
            scale = 1
        translation = muA - np.dot(scale*muB,rotation);  

        self.scale = scale
        self.rot = rotation
        self.trans = translation
        return scale,rotation,translation
        #A - (np.dot((scale * B),rotation) + translation)

    def get_error(self,data):
        A = data[:,0:self.dim]
        B = data[:,self.dim:]        
        diff = A - (np.dot(self.scale*B,self.rot) + self.trans)
        dist = np.apply_along_axis(np.linalg.norm,1,diff)
        return dist

def difference_of_gaussians(im, radius):
    sigma=0.44248120*radius+0.01654135
    return gaussian_filter(im, sigma) - gaussian_filter(im, 1.6*sigma)

def laplacian_of_gaussian(im, sigma):
    return laplace(gaussian_filter(im, sigma, mode = 'constant'), mode = 'constant')
    
def nonmaxsup(im, radius, threshold):
    '''
    Non-max Suppression Algorithm.  Returns an image with pixels that
    are not maximal within a square neighborhood zeroed out.
    '''
    
    # Normalize the image & threshold
    im = im / im.max()
    im[np.nonzero(im<threshold)] = 0

    # Extract local maxima by performing a grey scale morphological
    # dilation and then finding points in the corner strength image that
    # match the dilated image and are also greater than the threshold.
    from scipy.ndimage import morphology
    num_dimensions = len(im.shape)
    neighborhood_size = radius * np.ones(num_dimensions)

    mx = morphology.grey_dilation(im, footprint=np.ones(neighborhood_size), mode='constant', cval=0)
    return im * (im >= mx)

def refine_subpixel_positions_3d(im, putative_centers):
    '''
    Refine peak estimation to subpixel precision by fitting a 2nd
    degree polynomial in each of the three dimensions to the points
    immediately around the peak.  Since this routine fits 3 1D
    polynomials (one for each dimension), it makes the assumption that
    the peak is relatively symmetric around the true peak.  For our
    purposes, this is probably accurate enough.
    '''

    subpixel_putative_centers = putative_centers.copy().astype(np.float32)

    x = np.arange(-1, 2)   # creates [-1, 0, 1]
    y = np.zeros(3)
    for n in range(putative_centers.shape[0]):

        try:

            # X direction
            y[0] = im[putative_centers[n,0]-1, putative_centers[n,1], putative_centers[n,2]]
            y[1] = im[putative_centers[n,0]  , putative_centers[n,1], putative_centers[n,2]]
            y[2] = im[putative_centers[n,0]+1, putative_centers[n,1], putative_centers[n,2]]
            p = np.polyfit(x, y, deg = 2)
            delta = -p[1]/(2*p[0])
            subpixel_putative_centers[n,0] += delta

            # Y direction
            y[0] = im[putative_centers[n,0], putative_centers[n,1]-1, putative_centers[n,2]]
            y[1] = im[putative_centers[n,0], putative_centers[n,1]  , putative_centers[n,2]]
            y[2] = im[putative_centers[n,0], putative_centers[n,1]+1, putative_centers[n,2]]
            p = np.polyfit(x, y, deg = 2)
            delta = -p[1]/(2*p[0])
            subpixel_putative_centers[n,1] += delta

            # Z direction
            y[0] = im[putative_centers[n,0]-1, putative_centers[n,1], putative_centers[n,2]-1]
            y[1] = im[putative_centers[n,0]  , putative_centers[n,1], putative_centers[n,2]  ]
            y[2] = im[putative_centers[n,0]+1, putative_centers[n,1], putative_centers[n,2]+1]
            p = np.polyfit(x, y, deg = 2)
            delta = -p[1]/(2*p[0])
            subpixel_putative_centers[n,2] += delta

        except IndexError:
            pass # Skip over keypoints on the edge of the image

    return subpixel_putative_centers

def compute_transform(p1, p2, best_matches):
    import cvxpy, cvxpy.utils

    # How many good matches are there?
    num_bad_matches = sum([x == None for x in best_matches])
    num_good_matches = p1.shape[0]-num_bad_matches

    # Need at least three points for a good translation fit...
    if (num_good_matches < 3):
        print 'ERROR: not enough matches to compute a 3D affine fit.'
        exit(1)

    # Prepare data for fitting
    X = cvxpy.utils.ones((3, num_good_matches))
    Y = cvxpy.utils.ones((3, num_good_matches))
    count = 0
    for i in range(p1.shape[0]):
        if best_matches[i] != None:
            X[0,count] = p1[i,0]
            X[1,count] = p1[i,1]
            X[2,count] = p1[i,2]
            Y[0,count] = p2[best_matches[i],0]
            Y[1,count] = p2[best_matches[i],1]
            Y[2,count] = p2[best_matches[i],2]
            count += 1
            
    #    print X.T
    #    print Y.T
    
    translation = cvxpy.variable(3,1)

    cost_fn = cvxpy.norm1((X[:,1] + translation) - Y[:,1])
    for c in range(2, num_good_matches):
        cost_fn += cvxpy.norm1((X[:,c] + translation) - Y[:,c])
    
    p = cvxpy.program(
        cvxpy.minimize(cost_fn)
        )
    p.solve(quiet=True)
    
    return ((1,np.identity(3), np.array(translation.value).squeeze()), num_good_matches)

def compute_geometric_transform(p1,p2,best_matches):
    """
    Given two lists of euclidean points, solves for the
    transformation T that minimizes p1 - T(p2) in  a robust
    least squares sense.
    The transformation can be applied to a volume using
    the function transform_volume.
    """
    # How many good matches are there?
    num_bad_matches = sum([x == None for x in best_matches])
    num_good_matches = p1.shape[0]-num_bad_matches

    # Prepare data for fitting
    A = np.ones((3, num_good_matches))
    B = np.ones((3, num_good_matches))
    count = 0
    for i in range(p1.shape[0]):
        if best_matches[i] != None:
            A[0,count] = p1[i,0]
            A[1,count] = p1[i,1]
            A[2,count] = p1[i,2]
            B[0,count] = p2[best_matches[i],0]
            B[1,count] = p2[best_matches[i],1]
            B[2,count] = p2[best_matches[i],2]
            count += 1
    A = A.T
    B = B.T

    model = GeometricTransform(bScale=False)
    data = np.hstack((A,B))

    # Need at least seven points for a good transform fit...
    if (num_good_matches < 7):
        print 'WARNING: not enough matches to compute a geometric transform.'
        return 1, np.identity(3), np.array([0,0,0])
    elif (num_good_matches < 20):
        print 'WARNING: not enough matches to compute a robust fit.'
        return model.fit(data)
    else:
        import lflib.calibration.ransac as ransac
        try:
            bestdata = ransac.ransac(data,model,
                                     10, #rand samp size (num required to fit)
                                     30, #num iterations
                                     4.0, #transformed dist required to be considered inlier,
                                     15, #min inliers to be considered 
                                     debug=False,return_all=False)
            return model.fit(bestdata)
        except ValueError:
            return model.fit(data)

#----------------------------------------------------------------------------#
#                        INTEREST POINT FUNCTIONS
#----------------------------------------------------------------------------#

def ip_detect(im, sigma, neighborhood=3,threshold=0.3):
    # Compute the laplacian of gaussian filter for the image.
#    log_im = laplacian_of_gaussian(im, sigma)
    log_im = difference_of_gaussians(im, sigma)
                
    # Detect the local maxima in the log_im.  These are putative
    # interest points... that is, they are points that we are likely
    # to be able to detect and track in adjacent images.
    lf_nonmaxsup = nonmaxsup(log_im, neighborhood, threshold)  # Threshold of 0.3, nonmaxsup neighborhood of 3x3x3
    local_maxima = np.nonzero(lf_nonmaxsup)

    # Combine result into a Nx3 array so that it is easier to work with later on.
    local_maxima = np.array([local_maxima[0], local_maxima[1], local_maxima[2]]).T
    subpixel_local_maxima = refine_subpixel_positions_3d(log_im, local_maxima)

    return subpixel_local_maxima

def match_points(p1, p2, rejection_threshold):
    '''
    Compute the n^2 pairwise matches to associate points in p1 with
    their nearest neighbors in p2.
    '''
    best_matches_p1_to_p2 = []
    best_matches_p2_to_p1 = []

    # Find matches from p2 to p1
    for i in range(p2.shape[0]):
        diff = p1 - np.tile(p2[i,:], (p1.shape[0], 1))
        dist = np.apply_along_axis(np.linalg.norm, 1, diff)
        best_matches_p2_to_p1.append( np.argmin(dist) )

    # Find matches from p1 to p2
    for i in range(p1.shape[0]):
        diff = p2 - np.tile(p1[i,:], (p2.shape[0], 1))
        dist = np.apply_along_axis(np.linalg.norm, 1, diff)
        best_matches_p1_to_p2.append( np.argmin(dist) )

    # Pick out the matches for which p1->p2 matches p2->p1
    final_matches_p1_to_p2 = []
    for i in range(len(best_matches_p1_to_p2)):
        m1 = best_matches_p1_to_p2[i]
        m2 = best_matches_p2_to_p1[m1]
        if (m2 == i) and (np.linalg.norm(p1[i,:] - p2[m1,:]) < rejection_threshold):
            final_matches_p1_to_p2.append(m1)
        else:
            final_matches_p1_to_p2.append(None)

    return final_matches_p1_to_p2

def translate_volume(vol, translation, minval):
    from scipy.ndimage import shift
    return shift(vol, shift = -translation, mode='nearest', prefilter = False, order = 1)

def transform_volume(vol, scale, rotation, translation, minval):
    from scipy.ndimage import affine_transform
    from scipy.ndimage import shift
    # Rotation/scale/translation must convert output coords to input coords...
    # Note that translation and shift had to separated because affine_transform applies translation prior
    # to rotation which is incorrect. 
    vol_trans = affine_transform(vol, rotation*(1/scale), mode='nearest', prefilter=False, order=1)
    vol_trans = shift(vol_trans, shift=translation, mode='nearest', prefilter=False, order=1)
    return vol_trans

#----------------------------------------------------------------------------#
#                             IMAP FUNCTIONS
#----------------------------------------------------------------------------#

def compute_key_frames_imap(in_tuple):
    input_filename = in_tuple[0]
    frame_frequency = in_tuple[1]
    frame_offset = in_tuple[2]
    frames_to_avg = in_tuple[3]
    mask = in_tuple[4]

    print '\t    Generating key frames every %d frames starting at %d using a %d frame avg.' % (frame_frequency, frame_offset, frames_to_avg)

    # Open the file and prepare to read key frames
    input_file = h5py.File(input_filename,'r')
    vol_shape = input_file['vol_shape'][...]
    num_timepoints = input_file['timeseries_volume'].shape[0]    
    if mask and 'vol_mask' in input_file.keys():
        vol_mask = input_file['vol_mask'][...]
    elif mask:
        print 'WARNING: mask requested but not present.'

    # Compute anchor positions.
    start_positions = np.arange(frame_offset,num_timepoints-frames_to_avg,frame_frequency)
    end_positions = start_positions + frames_to_avg

    anchor_vols = {}
    for i in range(start_positions.shape[0]):
        print '\t    reading anchor %d / %d  [ %d:%d ]' % (i+1, start_positions.shape[0],
                                                           start_positions[i], end_positions[i])
        vol_vec = input_file['timeseries_volume'][start_positions[i]:end_positions[i], :]
        #vol_min = np.reshape(input_file['vol_min'][...], (vol_shape[2], vol_shape[1], vol_shape[0]), order='f')
        anchor_vec = vol_vec.mean(axis = 0)
        anchor_vols[i] = np.reshape(anchor_vec, (vol_shape[2], vol_shape[1], vol_shape[0]), order='f') #- vol_min
        if mask and 'vol_mask' in input_file.keys():
            anchor_vols[i] *= vol_mask

    input_file.close()
    return (num_timepoints, anchor_vols, vol_shape, start_positions)

def align_image_imap(in_tuple):
    input_filename = in_tuple[0]
    time_slice = in_tuple[1]
    sigma = in_tuple[2]
    reference_pts = in_tuple[3]
    ref_t = in_tuple[4]
    ref_scale = in_tuple[5]
    ref_rot = in_tuple[6]
    ref_trans = in_tuple[7]
    minval = in_tuple[8]
    mask = in_tuple[9]
    neighborhood = in_tuple[10]
    threshold = in_tuple[11]
    transonly = in_tuple[12]

    # STEP 1: INTEREST POINT DETECTION
    #
    input_file = h5py.File(input_filename,'r')
    vol_vec = input_file['timeseries_volume'][time_slice, :]
    vol_shape = input_file['vol_shape'][...]
    vol = np.reshape(vol_vec, (vol_shape[2], vol_shape[1], vol_shape[0]), order='f')
    #vol_min = np.reshape(input_file['vol_min'][...], (vol_shape[2], vol_shape[1], vol_shape[0]), order='f')
    temp_vol = vol.copy() #-vol_min
    if mask and 'vol_mask' in input_file.keys():
        vol_mask = input_file['vol_mask'][...]
        temp_vol*=vol_mask
    
    input_file.close()
    interest_pts = ip_detect(temp_vol, sigma, neighborhood, threshold)

    #STEP 2: FIND NEAREST KEYFRAME
    #
    keyndx = np.argmin(np.abs(np.array(ref_t) - time_slice))

    # STEP 3: INTEREST POINT MATCHING
    #
    best_matches = match_points(reference_pts[keyndx],
                                interest_pts, 10) # Rejection threshold = 50 px

    # STEP 3: RIGID (i.e. SIMILARITY) FITTING BETWEEN TO NEAREST KEYFRAME)
    #
    if transonly:
        (scale,rotation,translation) = compute_transform(reference_pts[keynex], interest_pts, best_matches)
    else:
        (scale,rotation,translation) = compute_geometric_transform(reference_pts[keyndx], interest_pts, best_matches)

    # STEP 4: COMBINE FRAME TRANSFORM WITH ALL PRIOR KEYFRAME TRANSFORMS
    if transonly:
        t_combined = Translation
        for key in xrange(keyndx-1,0-1,-1):
            t_combined += ref_trans[key]
    else:
        s_combined = 1
        r_combined = rotation*(1/scale)
        t_combined = translation
        for key in xrange(keyndx-1,0-1,-1):
            r_combined = np.dot(ref_rot[key]*(1/ref_scale[key]), r_combined)
            t_combined = np.dot(ref_rot[key]*(1/ref_scale[key]), t_combined) + ref_trans[key]           

    # STEP 5: TRANSFORM VOLUME TO MATCH FIRST KEYFRAME
    #
    if transonly:
        im_out = translate_volume(vol, t_combined, minval)
    else:
        im_out = transform_volume(vol, s_combined, r_combined, t_combined, minval)
    tform = (scale,rotation,translation)
        

    """
    # STEP 4: TRANSFORM VOLUME TO MATCH NEARSET KEYFRAME
    #
    if transonly:
        im_out = translate_volume(vol, translation, minval)
    else:
        im_out = transform_volume(vol, scale, rotation, translation, minval)
    tform = (scale,rotation,translation)

    # STEP 5: TRANSFORM BACK TO FIRST KEYFRAME
    for key in xrange(keyndx-1,0-1,-1):
        if transonly:
            im_out = translate_volume(im_out, ref_trans[key], minval)
        else:
            im_out = transform_volume(im_out, ref_scale[key], ref_rot[key], ref_trans[key], minval)
    """

    # Return the result
    return  (time_slice, interest_pts, best_matches, tform, im_out, keyndx)

if __name__ == '__main__':
    pass
