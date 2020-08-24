# =============================================================================
# This code sorts normal and inverted flash responses based on threshold values.
# It produces plots depicting the average and integrated responses of normal 
# flash ER and cytosolic responses.
# =============================================================================

import numpy
#from celltool.utility.path import path
import celltool.utility.datafile as datafile
#import glob
#import os
import matplotlib.pyplot as plt


def smooth(x,window_len=5,window='hanning'):
    """https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return(x)


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return(y)


fs = 3

ymin = -0.4
ymax = 0.6
ytmin = -0.4
ytmax = 0.61
yti = 0.2

yminer = -0.04
ymaxer = 0.04
ytminer = -0.04
ytmaxer = 0.05
ytier = 0.02

t = 0.1 #0.1 for flashes, 0.4 for gratings
T = numpy.arange(0,5,t) #5 for flashes, 31 for gratings

threshold = 0.14 #0.6 for flashes; 0.13 for sum>0
threshold_inverted = -0.1

colors = ['#9900cc','#00cc66','#808080']
alphas = [1,0.6]
brain_regions = ['M1','M5','M9-M10']

directory = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/2s_flashes/'
plot_dir = directory+'/plots/wo_screen'
m_dir = directory+'/measurements/'

for br in brain_regions[:]:
    
    header,rows = datafile.DataFile(m_dir+'average_responses-'+br+'-RGECO.csv').get_header_and_data()
    R = numpy.asarray(rows)
    R = numpy.asarray(R[:,6:],dtype = 'float')
    print(len(R)/2)
    
    header,rows = datafile.DataFile(m_dir+'average_responses-'+br+'-ER210.csv').get_header_and_data()
    ER = numpy.asarray(rows)
    ER = numpy.asarray(ER[:,6:],dtype = 'float')
    
    OFF = []
    ON = []
    OFFer = []
    ONer = []
    OFF_inverted = []
    ON_inverted = []
    OFFer_inverted = []
    ONer_inverted = []
    
    n=0
    while n<len(R):
        off = smooth(R[n,:])
        on = smooth(R[n+1,:])
        offer = smooth(ER[n,:])
        oner = smooth(ER[n+1,:])
        max_index_off = numpy.argmax(numpy.abs(off[10:30]))
        max_index_on = numpy.argmax(numpy.abs(on[10:30]))
#        print(off[max_index_off+10])
#        print(on[max_index_on+10])
        diff = on[max_index_on+10]-off[max_index_off+10]
#        print(diff)
        
        if diff>threshold:
            OFF.append(off[1:-3])
            ON.append(on[1:-3])
            OFFer.append(offer[1:-3])
            ONer.append(oner[1:-3])
        elif diff<threshold_inverted:
            OFF_inverted.append(off[1:-3])
            ON_inverted.append(on[1:-3])
            OFFer_inverted.append(offer[1:-3])
            ONer_inverted.append(oner[1:-3])
        n=n+2    
    
    OFF = numpy.asarray(OFF)
    ON = numpy.asarray(ON)
    print(len(OFF))
    
    OFFi = numpy.asarray(OFF_inverted)
    ONi = numpy.asarray(ON_inverted)
    #print(len(OFFi))
    
    OFFer = numpy.asarray(OFFer)
    ONer = numpy.asarray(ONer)
    print(len(OFFer))
    
    OFFeri = numpy.asarray(OFFer_inverted)
    ONeri = numpy.asarray(ONer_inverted)
    #print(len(OFFer_inverted))
    
    iOFF = numpy.sum(OFF[:,10:30],axis = 1)
    iON = numpy.sum(ON[:,10:30],axis = 1)
    #print(iON)
    print(numpy.amin(iON))
    print(numpy.amax(iOFF))
    iOFFer = numpy.sum(OFFer[:,10:30],axis = 1)
    #print(iOFFer)
    #print(numpy.amax(iOFFer))
    iONer = numpy.sum(ONer[:,10:30],axis = 1)
 
    
    # PLOT AND SAVE CYTOSOLIC AVERAGE RESPONSE
    plt.figure(figsize=(fs, fs))
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(ymin,ymax)
    plt.xlim(0,T[-1])
    plt.xticks(numpy.arange(0,T[-1],1), fontsize=14)
    plt.yticks(numpy.arange(ytmin, ytmax, yti), fontsize=14)
    plt.xlabel('seconds',fontsize = 16)
    plt.ylabel('DF/F',fontsize = 16)

    aOFF = numpy.average(OFF,axis = 0)
    stOFF = numpy.std(OFF,axis = 0)/numpy.sqrt(len(R)/2)
    plt.fill_between(T,aOFF-stOFF,aOFF+stOFF,color = colors[0],alpha=0.1) 
    plt.plot(T,aOFF,color = colors[0],alpha = 1)
    
    aON = numpy.average(ON,axis = 0)
    stON = numpy.std(ON,axis = 0)/numpy.sqrt(len(R)/2)
    plt.fill_between(T,aON-stON,aON+stON,color = colors[0],alpha=0.1)
    plt.plot(T,numpy.average(ON,axis = 0),color = colors[0],alpha = 0.5)
        
    plt.axvline(x=1, color=colors[2], linestyle='-.')
    plt.axvline(x=3, color=colors[2], linestyle='-.')
    plt.savefig(plot_dir+'/average_responses-'+br+'-RGECO.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    #PLOT AND SAVE AVERAGE ER RESPONSES
    plt.figure(figsize=(fs, fs))
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(yminer,ymaxer)
    plt.xlim(0,T[-1])
    plt.xticks(numpy.arange(0,T[-1],1), fontsize=14)
    plt.yticks(numpy.arange(ytminer, ytmaxer, ytier), fontsize=14)
    plt.xlabel('seconds',fontsize = 16)
    plt.ylabel('DF/F',fontsize = 16)
    
    aOFFer = numpy.average(OFFer,axis = 0)
    stOFFer = numpy.std(OFFer,axis = 0)/numpy.sqrt(len(R)/2)
    plt.fill_between(T,aOFFer-stOFFer,aOFFer+stOFFer,color = colors[1],alpha=0.1) 
    plt.plot(T,aOFFer,color = colors[1],alpha = 1)
    
    aONer = numpy.average(ONer,axis = 0)
    stONer = numpy.std(ONer,axis = 0)/numpy.sqrt(len(R)/2)
    plt.fill_between(T,aONer-stONer,aONer+stONer,color = colors[1],alpha=0.1)
    plt.plot(T,numpy.average(ONer,axis = 0),color = colors[1],alpha = 0.5)
        
    plt.axvline(x=1, color=colors[2], linestyle='-.')
    plt.axvline(x=3, color=colors[2], linestyle='-.')    
    plt.savefig(plot_dir+'/average_responses-'+br+'-ER210.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close()  
    
    #PLOT AND SAVE CYTOSOLIC INTEGRATED RESPONSES
    iON_iOFF = []
    iON_iOFF.append(iON)
    iON_iOFF.append(iOFF)
    iON_iOFF_stack = numpy.asarray(iON_iOFF)
    print(iON_iOFF_stack.shape)
    plt.figure(figsize=(fs,fs))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['ON','OFF'],fontsize = 16)
    plt.ylabel('cytosol [a.u.]',fontsize = 16)
    plt.yticks(fontsize=14)   
    
    parts = ax.violinplot(iON_iOFF,positions=[1,2], vert=True, showmeans=False,showextrema=False, showmedians=False)    
    for pc in parts['bodies']:
        pc.set_facecolor('#DDA0DD')
        pc.set_edgecolor(colors[0])
        pc.set_alpha(1)
    quartile1, medians, quartile3 = numpy.percentile(iON_iOFF, [25, 50, 75], axis=1)
    inds = numpy.arange(1, len(medians) + 1)
    
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color=colors[0], linestyle='solid', lw=5)
    ax.vlines(1, numpy.amin(iON), numpy.amax(iON), color=colors[0], linestyle='solid', lw=1)
    ax.vlines(2, numpy.amin(iOFF), numpy.amax(iOFF), color=colors[0], linestyle='solid', lw=1)

    plt.savefig(plot_dir+'/integrated_responses-'+br+'violinplt-RGECO.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close() 

    #PLOT AND SAVE ER INTEGRATED RESPONSES
    iONer_iOFFer = []
    iONer_iOFFer.append(iONer)
    iONer_iOFFer.append(iOFFer)
    iONer_iOFFer_stack = numpy.asarray(iONer_iOFFer)
    plt.figure(figsize=(fs,fs))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['ON','OFF'], fontsize = 16)
    plt.ylabel('ER [a.u.]',fontsize = 16)
    plt.yticks(fontsize=14)
    
    parts = ax.violinplot(iONer_iOFFer,positions=[1,2], vert=True, showmeans=False,showextrema=False, showmedians=False)    
    for pc in parts['bodies']:
        pc.set_facecolor('#99ff99')
        pc.set_edgecolor(colors[1])
        pc.set_alpha(1)
    quartile1, medians, quartile3 = numpy.percentile(iONer_iOFFer, [25, 50, 75], axis=1)
    inds = numpy.arange(1, len(medians) + 1)
    
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color=colors[1], linestyle='solid', lw=5)
    ax.vlines(1, numpy.amin(iONer), numpy.amax(iONer), color=colors[1], linestyle='solid', lw=1)
    ax.vlines(2, numpy.amin(iOFFer), numpy.amax(iOFFer), color=colors[1], linestyle='solid', lw=1)    
    
    plt.savefig(plot_dir+'/integrated_responses-'+br+'violinplt-ER210.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close()


