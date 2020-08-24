import numpy
from celltool.utility.path import path
import celltool.utility.datafile as datafile
import matplotlib.image as mpimg
from PIL import Image
import glob
import os

t = 0.1
bins = numpy.arange(0,4,t)  

"""Sets up some paths..."""
parent_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Tm3/flashes/lobula/'
parent_path = path('C:/Users/vymak_i7/Desktop/New_ER/Screen/Tm3/flashes/lobula/')

"""I'll explain this in person, ignore through line 54."""
header,rows = datafile.DataFile(parent_path/'directories.csv').get_header_and_data()

dir_list = [parent_dir+'/'+row[1] for row in rows]
path_list = [parent_path/row[1] for row in rows]

for directory,p in zip(dir_list[:],path_list[:]):
    
    print(directory)
    
    plot_dir = directory+'/plots/'
    print(plot_dir)
    
    image_dir1 = directory+'/RGECO'
    print(image_dir1)
    
    """This IDs and sorts image file names for one channel."""
    imagefiles1 = glob.glob(os.path.join(image_dir1,'*tif'))
    #sorted_files = sorted(imagefiles1, key=lambda x: int(x.split('.')[0]))
    #print(imagefiles1)
    # imagefiles1.sort(key=lambda f: int(filter(str.isdigit,f)))
    #print imagefiles
    
    
    """Loads images (for one channel)"""
    I1 = [mpimg.imread(imagefile) for imagefile in imagefiles1]
    print(I1[0].dtype)
    
    image_dir2 = directory+'/ER210'
    print(image_dir2)
    
    """This IDs and sorts image file names for one channel."""
    imagefiles2 = glob.glob(os.path.join(image_dir2,'*tif'))
    #imagefiles2.sort(key=lambda f: int(filter(str.isdigit,f)))
    #print imagefiles
    
    """Loads images (for one channel)"""
    I2 = [mpimg.imread(imagefile) for imagefile in imagefiles2]
    print(I2[0].dtype)
    
    stim_file = p.files('*-frames.csv')[0]
    
    header,rows = datafile.DataFile(stim_file).get_header_and_data()
    
    R = numpy.asarray(rows)
    
    im_num=0
    ER = R[R[:,1].argsort()]
    for b in bins:
        bi1 = numpy.searchsorted(ER[:,1],b)
        bi2 = numpy.searchsorted(ER[:,1],b+t)
        
        binned_images = [I1[int(im)-1] for im in ER[bi1:bi2,-1]]
        M = numpy.mean(binned_images,axis = 0)
        #print M.shape
        M = numpy.asarray(M,dtype = 'uint8')
        out = Image.fromarray(M)
        if im_num<10: 
            out.save(directory+'/binned_images-RGECO/000'+str(im_num)+'.tif')
        if im_num>=10 and im_num<100:
            out.save(directory+'/binned_images-RGECO/00'+str(im_num)+'.tif')
        if im_num>=100 and im_num<1000:
            out.save(directory+'/binned_images-RGECO/0'+str(im_num)+'.tif')
        elif im_num>1000:
            out.save(directory+'/binned_images-RGECO/'+str(im_num)+'.tif')
        #image.write_array_as_image_file(M,directory+'/RGECO_binned/'+str(stim_dir[i])+'-'+str(b)+'.tif')
        
        binned_images = [I2[int(im)-1] for im in ER[bi1:bi2,-1]]
        M = numpy.mean(binned_images,axis = 0)
        M = numpy.asarray(M,dtype = 'uint8')
        #print M.shape
        out = Image.fromarray(M)
        if im_num<10: 
            out.save(directory+'/binned_images-ER210/000'+str(im_num)+'.tif')
        if im_num>=10 and im_num<100:
            out.save(directory+'/binned_images-ER210/00'+str(im_num)+'.tif')
        if im_num>=100 and im_num<1000:
            out.save(directory+'/binned_images-ER210/0'+str(im_num)+'.tif')
        elif im_num>1000:
            out.save(directory+'/binned_images-ER210/'+str(im_num)+'.tif')
        #image.write_array_as_image_file(M,directory+'/RGECO_binned/RGECO_binned-'+str(b)+'.tif')
        
        im_num=im_num+1