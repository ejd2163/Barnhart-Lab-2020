#To import volumes from FAFB into tiff files  

from cloudvolume import CloudVolume
from tifffile import imwrite 
# Initialize volume with image data
vol = CloudVolume('precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig_sharded', use_https=True, fill_missing=True, mip=0)
# Define bounding box for the cut out
# This has to be in pixel not nanometer (1 px = 4 x 4 x 40nm)
bbox = [[130000, 130500],  # x min and x max 
        [44500, 45000],  # y min and y max 
        [1500, 1600]]   # z min and z max 
# Get the actual image data as array 
img = vol[bbox[0][0]:bbox[0][1],
          bbox[1][0]:bbox[1][1],
          bbox[2][0]:bbox[2][1]]
# The result is a 500 x 500 x 100 x 1 array of 0-255 values 
# Let's get rid of the last axis (this is irrelevant since it's an 8 bit image)
img = img[:, :, :, 0]
# Save to stacked Tiff (note the transpose via .T to bring the z axis in front)
imwrite('stacked.tiff', img.T)    