import numpy
from celltool.utility.path import path
import celltool.utility.datafile as datafile
import matplotlib.image as mpimg
from PIL import Image
import glob
import os
import scipy.ndimage as ndimage
from matplotlib import pylab

def smooth(x,window_len=5,window='hanning'):
    """https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

fs = 3
t = 0.2

ymin = -0.6
ymax = 0.6
ytmin = -0.6
ytmax = 0.61
yti = 0.2

T = numpy.arange(0,20,t) #(0,5,t) for flashes; (0,31,t) for gratings

header_out = ['sample','brain_region','ROI','x','y','epoch']
header_out.extend(list(T))
#print header_out

color = '#8F02C5' 
alphas = [1,0.6]
brain_regions = ['M1','M5','M9-M10']

parent_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/10s_flashes/'
parent_m_dir = parent_dir + '/measurements/'
parent_path = path('C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/10s_flashes/')
header,rows = datafile.DataFile(parent_path/'directories.csv').get_header_and_data()

dir_list = [parent_dir+'/'+row[1] for row in rows]
path_list = [parent_path/row[1] for row in rows]

for br in brain_regions[:]:
    O_DFF=[]
    for directory,p,row in zip(dir_list[:],path_list[:],rows[:]):
        OUT_DFF=[]
        print(directory)
        
        image_dir = directory+'/binned_images-RGECO'
        plot_dir = directory+'/plots/'
        m_dir = directory+'/measurements/'        
        imagefiles = glob.glob(os.path.join(image_dir,'*tif'))

        
        mask_file = directory+'/mask-'+br+'.tif'
        
        I = [mpimg.imread(imagefile) for imagefile in imagefiles]
        print(I[0].dtype)
    
        
        BG_mask = mpimg.imread(directory+'/background.tif')
        
        labels = ndimage.measurements.label(BG_mask)
        m = labels[0]==1
        IBG = I*m
        BG = [numpy.sum(ibg)/numpy.sum(m) for ibg in IBG]
        
        pylab.plot(BG)
        pylab.savefig(plot_dir+'background.tif')
        pylab.close()
        
        Ic = [im-bg for im,bg in zip(I,BG)]
        
        mask = mpimg.imread(mask_file)
        mask = mask.swapaxes(0,1)
        
        labels = ndimage.measurements.label(mask)
        L = labels[0].swapaxes(0,1)
        out = Image.fromarray(L)
        #out.save(directory+'/'+br+'_RGECO_labels.tif')
        print(labels[1])
        
        n=1
        while n<labels[1]+1:
        #while n==1:
            
            m = L==n
            centy,centx=ndimage.measurements.center_of_mass(m)
            #print centx
            #print centy
            
            IM = Ic*m
            A = [numpy.sum(im)/numpy.sum(m) for im in IM]
            
            pylab.figure(figsize=(fs, fs))
            ax = pylab.subplot(111)  
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            # pylab.ylim(ymin,ymax)
            pylab.xlim(0,T[-1])
            pylab.xticks(numpy.arange(0,T[-1],5), fontsize=12)
            # pylab.yticks(numpy.arange(ytmin, ytmax, yti), fontsize=12)
            pylab.xlabel('seconds',fontsize = 14)
            pylab.ylabel('DF/F',fontsize = 14)
            
            DFF=[row[1],br,n,centx,centy,'light']
            roi_A = A
            roi_A = numpy.asarray(roi_A)
            # roi_A = smooth(roi_A)
            # roi_A = numpy.asarray(roi_A[:-4])
            baseline = numpy.median(roi_A[:10])
            dff = (roi_A-baseline)/baseline
            DFF.extend(dff)
            pylab.plot(T,dff,color = color,alpha=alphas[1])
            OUT_DFF.append(DFF)
            O_DFF.append(DFF)
            
            pylab.savefig(plot_dir+'/'+br+'-ROI-'+str(n)+'-RGECO.png',dpi=300,bbox_inches = 'tight')
            #pylab.show()
            pylab.close()
            n=n+1
            
        datafile.write_data_file([header_out]+OUT_DFF,m_dir+'average_responses-'+br+'-RGECO.csv')

    
    datafile.write_data_file([header_out]+O_DFF,parent_m_dir+'average_responses-'+br+'-RGECO.csv')
    
