# Import modules
import os, shutil, sys
import subprocess
import numpy as np
import nibabel as nib
import glob
import time
import h5py

# Local imports
from wrappers import allineate,volreg, AFNItoNIFTI, generate_alignment_video
from lflib.imageio import load_image
from util.volume import volume_to_vector, volumes_to_vectors

#------------------------------------------------------------------------------
# Main alignment function

def motion_correct_volreg(base, frame, outfn):
    cmd = ['-zpad', '10']
    #cmd += ['-weight', base]
    cmd += ['-rot_thresh', '0.01']
    cmd += ['-x_thresh', '0.01']
    cmd.append('-base')
    cmd.append(base)
    #if options.motion_file is None:
    #    options.motion_file = os.path.splitext(nii_path)[0]+'_motion_parameters.txt'
    #cmd += ['-dfile', options.motion_file]
    #XX: add back motion fileee
    #cmd += ['-verbose']
    cmd += ['-prefix', outfn]
    cmd += [frame]
    volreg(cmd,debug=True)

def motion_correct_allineate(base, frame, outfn):
    #cmd = ['-zpad', '10']
    #cmd = ['-nmi']
    cmd = ['-ls']
    cmd += ['-onepass']
    cmd += ['-maxrot', '8']
    cmd += ['-maxshf', '8']
    cmd += ['-warp', 'shift_only']
    cmd += ['-conv', '0.01']
#    cmd += ['-automask+5']
    cmd += ['-final', 'cubic']
    #cmd += ['-allcost']
    #cmd += ['-source_automask']
    #cmd += ['-weight', base]
    #cmd +=['-wtprefix', outfn+'weight']
    #cmd += ['-rot_thresh', '0.01']
    #cmd += ['-x_thresh', '0.01']
    #cmd += ['-cubic']
    cmd.append('-base')
    cmd.append(base)
    #if options.motion_file is None:
    #    options.motion_file = os.path.splitext(nii_path)[0]+'_motion_parameters.txt'
    #cmd += ['-dfile', options.motion_file]
    #XX: add back motion fileee
    #cmd += ['-verbose']
    cmd += ['-prefix', outfn]

    cmd += [frame]
    #volreg(cmd,debug=True)
    allineate(cmd,debug=True)

def motion_correct(array4d, options, wait=60, 
                   base_path='/lfdata/tmp/', allineate=True):
    """
    Use AFNI's 3dvolreg function to do image alignment. 
    """
    # Make a local directory for temp files
    here_path = os.path.abspath(base_path)
    temp_dir = os.path.join(here_path,'temp_dir')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    # First, convert the array to NIFTI:
    print "  Converting array to NIFTI..."
    nii_path = array_to_nii(array4d, os.path.join(here_path, 
                                                  'temp_dir', 
                                                  'test_motion.nii'))
    input_format = nii_path + '[%d]'
    prefix_format = base_path + '/temp_dir/regframe%d'
    # 3dvolreg adds this suffix to the above prefix to get the filename
    volreg_suffix = '+tlrc.BRIK'
    output_format = prefix_format + '+tlrc.BRIK'
    weight_format = prefix_format + 'weight+tlrc.BRIK'
    base_frame = input_format % 0

    if allineate:
        motion_correct_frame = motion_correct_allineate
    else:
        motion_correct_frame = motion_correct_volreg

    if options.sequential:
        for t in xrange(array4d.shape[-1]):
            print 'TIME',t
            input_frame = input_format % t
            output_frame = prefix_format % t
            motion_correct_frame(base_frame, input_frame, output_frame)
            base_frame = output_frame + volreg_suffix
    else:
        motion_correct_frame(base_frame, nii_path, prefix_format % 0)

    # Third, wait for the file to write to disk
    # checking if it is there periodically.
#    file_name = base_path+'/temp_dir/volreg+tlrc.BRIK'
#    motion_correct_frame(input_fname, frame_fname, file_name)
    
    # Fourth, convert aligned results from AFNI to a numpy array.
    # Write h5 file.
    vol_shape = array4d.shape[:-1][::-1]
    num_t = array4d.shape[3]
    num_voxels = np.prod(vol_shape)
    output_filename = options.output_filename
    output_file = h5py.File(output_filename,'w')
    output_file.create_dataset('vol_shape', data=vol_shape)
    output_volume = output_file.create_dataset('timeseries_volume',
                                               shape=(num_t, num_voxels))
    if options.sequential:
        for t in xrange(num_t):
            #XXX: switch back to output_format
            #vol = afni_to_array(weight_format % t)
            vol = afni_to_array(output_format % t)
            vec = volume_to_vector(vol)
            output_volume[t, :] = vec
    else:
            vols = afni_to_array(output_format % 0)
            vecs = volumes_to_vectors(np.transpose(vols, (3,0,1,2)))
            output_volume[:] = vecs
    output_file.flush()
    if options.stats:
        print '  Updating time series stats.'
        from util.timeseries import update_timeseries_stats
        update_timeseries_stats(output_filename)
    # Clean up temp files
    try:
        shutil.rmtree(os.path.join(base_path,'temp_dir'))
    except:
        print "Check to make sure files in", os.path.join(base_path,'temp_dir'), "are gone."
    print "  Done aligning time-series volume with 3dvolreg (AFNI)."

    return output_volume

#-------------------------------------------------------------------------------
# Conversion functions

def array_to_nii(array_4d, outfile):
    """
    Convert a 4d numpy array with dimensions (y,x,z,t) into 
    an NIFTI file readable by AFNI using Nibabel. 
    """
    img = nib.Nifti1Image(array_4d, np.eye(4))
    img.to_filename(outfile)

    return os.path.abspath(outfile)

def afni_to_array(input_path, wait=20):
    """
    Convert an AFNI BRIK into a numpy array. 
    """
    # Call wrapper for AFNI's 3dAFNItoNIFTI to convert the AFNI BRIK file
    # specified in 'input_path' into a NIFTI file.
    temp_file = os.path.join(os.path.split(input_path)[0],'volreg.nii')
    cmd = ['-prefix']
    cmd += [os.path.splitext(temp_file)[0]]
    cmd += [input_path]

    AFNItoNIFTI(cmd)

    # Load the NIFTI file using Nibabel
    try:
        img = nib.load(temp_file)
    except:
        print ("If NIFTI file could not be read, you may have to "
               "lengthen the 'wait' parameter to allow time for everything "
               "to be written to disk (this is most likely for large data).")
    
    # Get the array data        
    array_data = img.get_data()

    # Clean up temp file
    subprocess.Popen(['rm',temp_file]).wait()

    return array_data

def nii_to_array(input_path, wait=20):
    """
    Convert a NIFTI file into a numpy array. 
    """
    # Load the NIFTI file using Nibabel
    try:
        img = nib.load(input_path)
    except:
        raise Exception("Could not load NIFTI file.")
    
    # Get the array data        
    array_data = img.get_data()

    return array_data

def h5_to_nii(infile, outfile, chunks=1):
    """
    Convert an HDF5 file output by LFAnalyze (with dimensions (y,x,z,t)) into 
    an NIFTI file readable by AFNI and arranged (x,y,z,t) using array_to_nii.
    """
    import h5py
    input_data = h5py.File(infile, 'r')

    ts_vol = input_data['timeseries_volume']
    vol_shape = input_data['vol_shape']

    out = []
    ntimesteps = ts_vol.shape[0]
    input_chunks = np.array_split(np.arange(ntimesteps), chunks)
    for idx, chunk in enumerate(input_chunks):
        if chunk.size > 0:
            X = ts_vol[chunk,...]
            Y = np.reshape(X, (chunk.size, vol_shape[2], vol_shape[1], 
                               vol_shape[0]), order='f')
            Y = np.transpose(Y,(1,2,3,0))
            outpath, outfilename = os.path.split(outfile)
            array_to_nii(Y, outpath+'/'+outfilename.split('.')[0]+
                         '_'+str(idx)+'.nii')
            X = 0
            Y = 0

def nii_to_h5(infile, verbose=True):
    """
    Convert a NIFTI file arranged (x,y,z,t) into an HDF5 file 
    for LFAnalyze (with dimensions (t,y,x,z) flattened).
    """
    try:
        np_arr = nii_to_array(infile)
    except:
        np_arr = nii_to_array(infile)
    vol_shape = (np_arr.shape[2], np_arr.shape[1], np_arr.shape[0]) # (z,x,y)
    Z = np.transpose(np_arr, (3,0,1,2))
    Z = np.reshape(Z, (Z.shape[0], np.prod(Z.shape[1:4])),order='f')
    import h5py
    outfilename = os.path.splitext(infile)[0]+'.h5'
    output_file = h5py.File(outfilename,'w')
    output_file.create_dataset('vol_shape', data=vol_shape)
    output_file.create_dataset('timeseries_volume', data=Z)
    output_file.close()

def nii_to_tiff(infile, outfile):
    """
    Convert a NIFTI file arranged (x,y,z,t) into an HDF5 file 
    for LFAnalyze (with dimensions (t,y,x,z) flattened).
    """
    # Make sure the output directory exists
    print os.path.dirname(outfile)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    
    try:
        np_arr = nii_to_array(infile)
    except:
        np_arr = nii_to_array(infile)
    vol_shape = (np_arr.shape[2], np_arr.shape[1], np_arr.shape[0]) # (z,x,y)
    Z = np.transpose(np_arr, (3,0,1,2))
    Z = np.reshape(Z, (Z.shape[0], np.prod(Z.shape[1:4])),order='f')

    import console_output as co
    print "Convert TIF images"
    from lflib.imageio import save_image
    for i in range(Z.shape[0]):
        co.simple_progress_bar(i, Z.shape[0])

        vol = np.reshape(Z[i,:],
                         (vol_shape[2], vol_shape[1], vol_shape[0]), 
                         order='f')
        filename = outfile % (i)
        save_image(filename, vol)
        
def tif_to_nii(infile, outfile):
    """
    Convert an HDF5 file output by LFAnalyze (with dimensions (y,x,z,t)) into 
    an NIFTI file readable by AFNI and arranged (x,y,z,t) using array_to_nii.

    """
    X = load_image(infile)
    array_to_nii(X, outfile)
    return outfile
