import numpy
from celltool.utility.path import path
import celltool.utility.datafile as datafile
import matplotlib.image as mpimg
from PIL import Image
import glob
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

fs = 3
t = 0.1

ymin = -0.2
ymax = 1.0
ytmin = 0.
ytmax = 1.01
yti = 0.2

yminer = -0.2
ymaxer = 0.6
ytminer = 0
ytmaxer = 0.61
ytier = 0.2

T = numpy.arange(0,5,t)

header_out = ['sample','br','ROI','x','y','epoch']
header_out.extend(list(T))
#print header_out

indices = [[0,50],[50,100],[100,150],[150,200]]
ENUM=[1,2,3,4]
colors = ['#8F02C5','#02C54F','#0F37FF','#F8A51B'] #purple, green, blue, orange
brain_regions = ['M1','M5','M9-M10']

parent_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/moving_light_bars/'
parent_path = path('C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/moving_light_bars/')
mdir = parent_dir+'/measurements/2s/'
header,rows = datafile.DataFile(parent_path/'2s_directories.csv').get_header_and_data()

dir_list = [parent_dir+'/'+row[1] for row in rows]
path_list = [parent_path/row[1] for row in rows]


for br in brain_regions:
    
    OUT_DFF=[] #RGECO

    for directory,p,row in zip(dir_list[:],path_list[:],rows):
        
        print(directory)
        if not os.path.exists(directory+'/plots/time'):
        	os.makedirs(directory+'/plots/time')
        if not os.path.exists(directory+'/plots/position'):
        	os.makedirs(directory+'/plots/position')
        
        image_dir = directory+'/binned_images-RGECO'
        plot_dir = directory+'/plots/time/'
        
        imagefiles = glob.glob(os.path.join(image_dir,'*tif'))
        
        mask_file = directory+'/mask-'+br+'.tif'
        
        I = [mpimg.imread(imagefile) for imagefile in imagefiles]
        print(I[0].dtype)
        
        BG_mask = mpimg.imread(directory+'/background.tif')
        
        labels = ndimage.measurements.label(BG_mask)
        m = labels[0]==1
        IBG = I*m
        BG = [numpy.sum(ibg)/numpy.sum(m) for ibg in IBG]        
        
        plt.plot(BG)
        plt.savefig(plot_dir+'background.tif')
        plt.close()
        
        Ic = [im-bg for im,bg in zip(I,BG)]
        
        mask = mpimg.imread(mask_file)
        mask = mask.swapaxes(0,1)
        
        labels = ndimage.measurements.label(mask)
        L = labels[0].swapaxes(0,1)
        out = Image.fromarray(L)
        out.save(directory+'/labels.tif')
        print(labels[1])
        
        n = 1
        while n<labels[1]+1:
        #while n==1:
            
            m = L==n
            centy,centx=ndimage.measurements.center_of_mass(m)
            #print centx
            #print centy
            
            IM = Ic*m
            A = [numpy.sum(im)/numpy.sum(m) for im in IM] #RGECO
            
            #plotting RGECO
            plt.figure(figsize=(fs, fs))
            ax = plt.subplot(111)  
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            # plt.ylim(ymin,ymax)
            plt.xlim(0,T[-1])
            plt.xticks(numpy.arange(0,T[-1],1), fontsize=12)
            # plt.yticks(numpy.arange(ytmin, ytmax, yti), fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('seconds',fontsize = 14)
            plt.ylabel('DF/F',fontsize = 14)
            for i,c,epoch_num in zip(indices,colors,ENUM):
                DFF=[row[1],br,n,centx,centy,epoch_num]
                roi_A = A[i[0]:i[1]]
                baseline = numpy.median(roi_A[:10])
                dff = (roi_A-baseline)/baseline
                DFF.extend(dff)
                ax.plot(T,dff,color = c)
                OUT_DFF.append(DFF)
            plt.savefig(plot_dir+'/'+br+'-RGECO-'+str(n)+'.png',dpi=300,bbox_inches = 'tight')
            # plt.show()
            plt.close()
            
            n=n+1
    
    print(br+' MLB count: '+str(len(OUT_DFF)/4))
    
    datafile.write_data_file([header_out]+OUT_DFF,mdir+'/average_responses-'+br+'.csv')
