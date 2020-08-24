import numpy
from celltool.utility.path import path
import celltool.utility.datafile as datafile
#import matplotlib.image as mpimg
#from PIL import Image
# import glob
# import os
#import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

fs = 3.0

subfont = 12
supfont = 16

xo = numpy.arange(0,240,0.3)
x = numpy.arange(0,241,30)
xti = numpy.arange(0,241,30)

#t = 0.2855
#bins = numpy.arange(0,29.95,t)
brain_regions = ['Lo2','Lo4']

color = ['#9900cc','#00cc66','#808080'] 

parent_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Tm3/gratings/lobula/'
parent_path = path('C:/Users/vymak_i7/Desktop/New_ER/Screen/Tm3/gratings/lobula/')
header,rows = datafile.DataFile(parent_path/'directories.csv').get_header_and_data()
parent_mdir = parent_dir+'/measurements/'

dir_list = [parent_dir+'/'+row[1] for row in rows]
path_list = [parent_path/row[1] for row in rows]


for br in brain_regions[:]:
    
    out = []
    out2 = []

    #import gratings ER and RGECO data and separate headers and rows from each dataset
    header,rows = datafile.DataFile(parent_mdir+'all_responses-'+br+'-ER210.csv').get_header_and_data()
    out = numpy.asarray(rows)
    out = numpy.asarray(out[:,3:],dtype = 'float')
    
    header2,rows2 = datafile.DataFile(parent_mdir+'all_responses-'+br+'-RGECO.csv').get_header_and_data()
    out2 = numpy.asarray(rows2)
    out2 = numpy.asarray(out2[:,3:],dtype = 'float')
    
    print(len(out))
    print(len(out2))
    
    M1 = numpy.mean(out,axis=0)
    baseline1 = numpy.median(M1[:18])
    WB1 = (M1-baseline1)/baseline1
    STD1 = numpy.std(out,axis=0)
    STE1 = STD1/numpy.sqrt(len(M1))
    
    M2 = numpy.mean(out2,axis=0)
    baseline2 = numpy.median(M2[:18])
    WB2 = (M2-baseline2)/baseline2
    STD2 = numpy.std(out2,axis=0)
    STE2 = STD2/numpy.sqrt(len(M2))
    
    #Plotting ER210 mean global response
    ymin = -0.6
    ymax = 0.6
    ytmin = -0.6
    ytmax = 0.61
    yti = 0.2
    plt.figure(figsize=(fs*1.75, fs))
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(ymin,ymax)
    plt.xlim(0,header[-1])
    plt.xticks(x,xti, fontsize=subfont)
    plt.yticks(numpy.arange(ytmin, ytmax, yti), fontsize=subfont)
    plt.xlabel('seconds',fontsize = subfont)
    plt.ylabel('DF/F',fontsize = subfont)
    # plt.title(br+' ER210',fontsize = supfont)
    
    plt.fill_between(xo,WB1+STE1,WB1-STE1,color = color[1],alpha = 0.2)
    plt.plot(xo,WB1,linewidth = 1.75, color = color[1])
    plt.savefig(parent_dir+'plots/'+br+'-all_responses-ER210.png',bbox_inches = 'tight',dpi=300)
    plt.show()
    plt.close()
    
    #Plotting RGECO mean global response
    ymin = -0.4
    ymax = 0.6
    ytmin = -0.4
    ytmax = 0.61
    yti = 0.2
    plt.figure(figsize=(fs*1.75, fs))
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(ymin,ymax)
    plt.xlim(0,header2[-1])
    plt.xticks(x,xti, fontsize=subfont)
    plt.yticks(numpy.arange(ytmin, ytmax, yti), fontsize=subfont)
    plt.xlabel('seconds',fontsize = subfont)
    plt.ylabel('DF/F',fontsize = subfont)
    # plt.title(br+' RGECO',fontsize = supfont)
    
    plt.fill_between(xo,WB2+STE2,WB2-STE2,color = color[0],alpha = 0.2)
    plt.plot(xo,WB2,linewidth = 1.75, color = color[0])
    plt.savefig(parent_dir+'plots/'+br+'-all_responses-RGECO.png',bbox_inches = 'tight',dpi=300)
    plt.show()
    plt.close()
    