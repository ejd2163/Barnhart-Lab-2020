# import relevant modules
import numpy as np
import celltool.utility.datafile as datafile
import matplotlib.pyplot as plt

fs = 3.0 #figure size

#flashes plot axis settings
ymin = -0.2 #cytosol
ymax = 0.6
ytmin = -0.2
ytmax = 0.61
yti = 0.2

yminer = -0.06 #ER
ymaxer = 0.06
ytminer = -0.06
ytmaxer = 0.061
ytier = 0.02

#gratings plot axis settings
yming = -0.2 #cytosol
ymaxg = 0.4
ytming = -0.2
ytmaxg = 0.41
ytig = 0.1

yminerg = -0.08 #ER
ymaxerg = 0.04
ytminerg = -0.08
ytmaxerg = 0.041
ytierg = 0.02

threshold = 1 #10 or 1
threshold_inverted = 1 #2 or 1

t = 0.2 #flashes
T = np.arange(0,20,t)

# tg = 0.2855 #gratings
# bins = np.arange(0,30,tg) 
# bins = bins[:-1]

colors = ['#9900cc','#00cc66','#808080'] #purple, green, gray 
alphas = [1.0, 0.5, 0.1]
brain_regions = ['M1','M5','M9-M10'] #relevant compartment regions

#moving light bars directory name
mlb_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/moving_light_bars/'
mlb_mdir= mlb_dir+'/measurements/10s'
map_mdir=mlb_mdir+'/mappable/'
unmap_mdir=mlb_mdir+'/unmappable/'

#input flashes directory name
flashes_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/10s_flashes/'
f_mdir = flashes_dir + '/measurements/'
f_pdir = flashes_dir +'/plots/'

# #input gratings directory name
# gratings_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/gratings/'
# g_mdir = gratings_dir + '/measurements/'
# g_pdir = gratings_dir +'/plots/'

for br in brain_regions[:]:
    
    #import flash ER and RGECO data w/ mapped RF centers and separate headers and rows from each dataset
    header,rows = datafile.DataFile(f_mdir+'mapped/newavg_flash-'+br+'-ER210.csv').get_header_and_data()
    flash_ER = np.asarray(rows)
    mapf_ER = np.asarray(flash_ER[:,6:],dtype = 'float')
    
    header,rows = datafile.DataFile(f_mdir+'mapped/newavg_flash-'+br+'-RGECO.csv').get_header_and_data()
    flash_CYT = np.asarray(rows)
    mapf_CYT = np.asarray(flash_CYT[:,6:],dtype = 'float')
    print(len(mapf_CYT))
    
    # #import flash ER and RGECO data w/ unmapped RF centers and separate headers and rows from each dataset
    # header,rows = datafile.DataFile(f_mdir+'unmapped/newavg_flash-'+br+'-ER210.csv').get_header_and_data()
    # flash_ER = np.asarray(rows)
    # umapf_ER = np.asarray(flash_ER[:,6:],dtype = 'float')
    
    # header,rows = datafile.DataFile(f_mdir+'unmapped/newavg_flash-'+br+'-RGECO.csv').get_header_and_data()
    # flash_CYT = np.asarray(rows)
    # umapf_CYT = np.asarray(flash_CYT[:,6:],dtype = 'float')
    # print(len(umapf_CYT)/2)
    
    # #import gratings ER and RGECO data w/ mapped RF centers and separate headers and rows from each dataset
    # header,rows = datafile.DataFile(map_mdir+'newavg_gDFF-'+br+'-ER210.csv').get_header_and_data()
    # gratings_ER = np.asarray(rows)
    # mapg_ER = np.asarray(gratings_ER[:,3:],dtype = 'float')
    
    # header,rows = datafile.DataFile(map_mdir+'newavg_gDFF-'+br+'-RGECO.csv').get_header_and_data()
    # gratings_CYT = np.asarray(rows)
    # mapg_CYT = np.asarray(gratings_CYT[:,3:],dtype = 'float')
    # print(len(mapg_CYT))
    
    # #import gratings ER and RGECO data w/ unmapped RF centers and separate headers and rows from each dataset
    # header,rows = datafile.DataFile(unmap_mdir+'newavg_gDFF-'+br+'-ER210.csv').get_header_and_data()
    # gratings_ER = np.asarray(rows)
    # umapg_ER = np.asarray(gratings_ER[:,3:],dtype = 'float')
    
    # header,rows = datafile.DataFile(unmap_mdir+'newavg_gDFF-'+br+'-RGECO.csv').get_header_and_data()
    # gratings_CYT = np.asarray(rows)
    # umapg_CYT = np.asarray(gratings_CYT[:,3:],dtype = 'float')
    # print(len(umapg_CYT))

    #separate responses w/ mapped RF centers into light and dark responses for ER and CYT
    mfl_CYT=[]
    mfd_CYT=[]
    mfl_ER=[]
    mfd_ER=[]
    f=0
    while f<len(mapf_CYT): 
        # mfd_CYT.append(mapf_CYT[f,:])
        mfl_CYT.append(mapf_CYT[f,:])
        # mfd_ER.append(mapf_ER[f,:])
        mfl_ER.append(mapf_ER[f,:])
        
        f=f+1
        
    # mfd_CYT = np.asarray(mfd_CYT, dtype='float') #convert strings to arrays
    mfl_CYT = np.asarray(mfl_CYT, dtype='float')
    # mfd_ER = np.asarray(mfd_ER, dtype='float')
    mfl_ER = np.asarray(mfl_ER, dtype='float')    
    
    # #separate responses w/ unmapped RF centers into light and dark responses for ER and CYT
    # umfl_CYT=[]
    # umfd_CYT=[]
    # umfl_ER=[]
    # umfd_ER=[]
    # f=0
    # while f<len(umapf_CYT)-1: 
    #     umfd_CYT.append(umapf_CYT[f,:])
    #     umfl_CYT.append(umapf_CYT[f+1,:])
    #     umfd_ER.append(umapf_ER[f,:])
    #     umfl_ER.append(umapf_ER[f+1,:])
        
    #     f=f+2
        
    # umfd_CYT = np.asarray(umfd_CYT, dtype='float') #convert strings to arrays
    # umfl_CYT = np.asarray(umfl_CYT, dtype='float')
    # umfd_ER = np.asarray(umfd_ER, dtype='float')
    # umfl_ER = np.asarray(umfl_ER, dtype='float') 
     

    #PLOT

    #mapped CYT flash responses
    plt.figure(figsize=(1.5*fs, fs))
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(ymin,ymax)
    plt.yticks(np.arange(ytmin, ytmax, yti), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,T[-1])
    plt.xticks(np.arange(0,T[-1],5), fontsize=14)
    plt.xlabel('seconds',fontsize = 16)
    plt.ylabel('DF/F',fontsize = 16)
    
    # aOFF = np.average(mfd_CYT,axis = 0)
    # semOFF = np.std(mfd_CYT,axis = 0)/np.sqrt(len(mfd_CYT))
    # plt.fill_between(T,aOFF-semOFF,aOFF+semOFF,color = colors[0],alpha=alphas[2]) 
    # plt.plot(T,aOFF,color = colors[0],alpha = alphas[0])
    
    aON = np.average(mfl_CYT,axis = 0)
    semON = np.std(mfl_CYT,axis = 0)/np.sqrt(len(mfl_CYT))
    plt.fill_between(T,aON-semON,aON+semON,color = colors[0],alpha=alphas[2])
    plt.plot(T,aON,color = colors[0])
        
    plt.axvline(x=5, color=colors[2], linestyle='-.')
    plt.axvline(x=15, color=colors[2], linestyle='-.')
    plt.savefig(f_pdir+'/mapped/'+br+'-average_responses-RGECO.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    #mapped ER flash responses
    plt.figure(figsize=(1.5*fs, fs))
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(yminer,ymaxer)
    plt.xlim(0,T[-1])
    plt.xticks(np.arange(0,T[-1],5), fontsize=14)
    plt.yticks(np.arange(ytminer, ytmaxer, ytier), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('seconds',fontsize = 16)
    plt.ylabel('DF/F',fontsize = 16)
    
    # aOFFer = np.average(mfd_ER,axis = 0)
    # stOFFer = np.std(mfd_ER,axis = 0)/np.sqrt(len(mfd_ER))
    # plt.fill_between(T,aOFFer-stOFFer,aOFFer+stOFFer,color = colors[1],alpha=alphas[2]) 
    # plt.plot(T,aOFFer,color = colors[1],alpha = alphas[0])
    
    aONer = np.average(mfl_ER,axis = 0)
    stONer = np.std(mfl_ER,axis = 0)/np.sqrt(len(mfl_ER))
    plt.fill_between(T,aONer-stONer,aONer+stONer,color = colors[1],alpha=alphas[2])
    plt.plot(T,aONer,color = colors[1])
        
    plt.axvline(x=5, color=colors[2], linestyle='-.')
    plt.axvline(x=15, color=colors[2], linestyle='-.')
    plt.savefig(f_pdir+'/mapped/'+br+'-average_responses-ER210.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close()   
    
    # #unmapped CYT flash responses 
    # plt.figure(figsize=(fs, fs))
    # ax = plt.subplot(111)  
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # plt.ylim(ymin,ymax)
    # plt.xlim(0,T[-1])
    # plt.xticks(np.arange(0,T[-1],1), fontsize=14)
    # plt.yticks(np.arange(ytmin, ytmax, yti), fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('seconds',fontsize = 16)
    # plt.ylabel('DF/F',fontsize = 16)
    
    # aOFF = np.average(umfd_CYT,axis = 0)
    # semOFF = np.std(umfd_CYT,axis = 0)/np.sqrt(len(umfd_CYT))
    # plt.fill_between(T,aOFF-semOFF,aOFF+semOFF,color = colors[0],alpha=alphas[2]) 
    # plt.plot(T,aOFF,color = colors[0],alpha = alphas[0])
    
    # aON = np.average(umfl_CYT,axis = 0)
    # semON = np.std(umfl_CYT,axis = 0)/np.sqrt(len(umfl_CYT))
    # plt.fill_between(T,aON-semON,aON+semON,color = colors[0],alpha=alphas[2])
    # plt.plot(T,aON,color = colors[0],alpha = alphas[1])
        
    # plt.axvline(x=5, color=colors[2], linestyle='-.')
    # plt.axvline(x=15, color=colors[2], linestyle='-.')
    # plt.savefig(f_pdir+'/unmapped/'+br+'-average_responses-RGECO.png',dpi=300,bbox_inches = 'tight')
    # plt.show()
    # plt.close()
    
    # #unmapped ER flash responses 
    # plt.figure(figsize=(fs, fs))
    # ax = plt.subplot(111)  
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # plt.ylim(yminer,ymaxer)
    # plt.xlim(0,T[-1])
    # plt.xticks(np.arange(0,T[-1],1), fontsize=14)
    # plt.yticks(np.arange(ytminer, ytmaxer, ytier), fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('seconds',fontsize = 16)
    # plt.ylabel('DF/F',fontsize = 16)
    
    # aOFFer = np.average(umfd_ER,axis = 0)
    # stOFFer = np.std(umfd_ER,axis = 0)/np.sqrt(len(umfd_ER))
    # plt.fill_between(T,aOFFer-stOFFer,aOFFer+stOFFer,color = colors[1],alpha=alphas[2]) 
    # plt.plot(T,aOFFer,color = colors[1],alpha = alphas[0])
    
    # aONer = np.average(umfl_ER,axis = 0)
    # stONer = np.std(umfl_ER,axis = 0)/np.sqrt(len(umfl_ER))
    # plt.fill_between(T,aONer-stONer,aONer+stONer,color = colors[1],alpha=alphas[2])
    # plt.plot(T,aONer,color = colors[1],alpha = alphas[1])
        
    # plt.axvline(x=5, color=colors[2], linestyle='-.')
    # plt.axvline(x=15, color=colors[2], linestyle='-.')
    # plt.savefig(f_pdir+'/unmapped/'+br+'-average_responses-ER210.png',dpi=300,bbox_inches = 'tight')
    # plt.show()
    # plt.close() 
    
    # #integrated flash ER and CYT responses of ROIs w/ mapped RFs 
    # iOFF = np.sum(mfd_CYT[:,10:30],axis = 1)
    # iON = np.sum(mfl_CYT[:,10:30],axis = 1)
    # iOFFer = np.sum(mfd_ER[:,10:30],axis = 1)
    # iONer = np.sum(mfl_ER[:,10:30],axis = 1)

    # iON_iOFF = []
    # iON_iOFF.append(iON)
    # iON_iOFF.append(iOFF)
    # iON_iOFF_stack = np.asarray(iON_iOFF)
    # plt.figure(figsize=(fs,fs))
    # ax = plt.subplot(111)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.set_xticks([1,2])
    # ax.set_xticklabels(['ON','OFF'],fontsize = 16)
    # plt.ylabel('cytosol [a.u.]',fontsize = 16)
    # plt.yticks(fontsize=14)   
    
    # parts = ax.violinplot(iON_iOFF,positions=[1,2], vert=True, showmeans=False,showextrema=False, showmedians=False)    
    # for pc in parts['bodies']:
    #     pc.set_facecolor('#DDA0DD')
    #     pc.set_edgecolor(colors[0])
    #     pc.set_alpha(1)
    # quartile1, medians, quartile3 = np.percentile(iON_iOFF, [25, 50, 75], axis=1)
    # inds = np.arange(1, len(medians) + 1)
    
    # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=7)
    # ax.vlines(inds, quartile1, quartile3, color=colors[0], linestyle='solid', lw=9)
    # ax.vlines(1, np.amin(iON), np.amax(iON), color=colors[0], linestyle='solid', lw=1)
    # ax.vlines(2, np.amin(iOFF), np.amax(iOFF), color=colors[0], linestyle='solid', lw=1)

    # plt.savefig(f_pdir+'/mapped/'+br+'-integrated_responses-RGECO.png',dpi=300,bbox_inches = 'tight')
    # plt.show()
    # plt.close()     

    # #plot and save integrated ER flash responses (violin plots)
    # iONer_iOFFer = []
    # iONer_iOFFer.append(iONer)
    # iONer_iOFFer.append(iOFFer)
    # iONer_iOFFer_stack = np.asarray(iONer_iOFFer)
    # plt.figure(figsize=(fs,fs))
    # ax = plt.subplot(111)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.set_xticks([1,2])
    # ax.set_xticklabels(['ON','OFF'], fontsize = 16)
    # plt.ylabel('ER [a.u.]',fontsize = 16)
    # plt.yticks(fontsize=14)
    
    # parts = ax.violinplot(iONer_iOFFer,positions=[1,2], vert=True, showmeans=False,showextrema=False, showmedians=False)    
    # for pc in parts['bodies']:
    #     pc.set_facecolor('#99ff99')
    #     pc.set_edgecolor(colors[1])
    #     pc.set_alpha(1)
    # quartile1, medians, quartile3 = np.percentile(iONer_iOFFer, [25, 50, 75], axis=1)
    # inds = np.arange(1, len(medians) + 1)
    
    # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=7)
    # ax.vlines(inds, quartile1, quartile3, color=colors[1], linestyle='solid', lw=9)
    # ax.vlines(1, np.amin(iONer), np.amax(iONer), color=colors[1], linestyle='solid', lw=1)
    # ax.vlines(2, np.amin(iOFFer), np.amax(iOFFer), color=colors[1], linestyle='solid', lw=1)    
    
    # plt.savefig(f_pdir+'/mapped/'+br+'-integrated_responses-ER210.png',dpi=300,bbox_inches = 'tight')
    # plt.show()
    # plt.close()
    
    # # #integrated flash ER and CYT responses of ROIs w/ unmapped RFs 
    # # iOFF = np.sum(umfd_CYT[:,10:30],axis = 1)
    # # iON = np.sum(umfl_CYT[:,10:30],axis = 1)
    # # iOFFer = np.sum(umfd_ER[:,10:30],axis = 1)
    # # iONer = np.sum(umfl_ER[:,10:30],axis = 1)

    # # iON_iOFF = []
    # # iON_iOFF.append(iON)
    # # iON_iOFF.append(iOFF)
    # # iON_iOFF_stack = np.asarray(iON_iOFF)
    # # plt.figure(figsize=(fs,fs))
    # # ax = plt.subplot(111)
    # # ax.spines["top"].set_visible(False)
    # # ax.spines["right"].set_visible(False)
    # # ax.set_xticks([1,2])
    # # ax.set_xticklabels(['ON','OFF'],fontsize = 16)
    # # plt.ylabel('cytosol [a.u.]',fontsize = 16)
    # # plt.yticks(fontsize=14)   
    
    # # parts = ax.violinplot(iON_iOFF,positions=[1,2], vert=True, showmeans=False,showextrema=False, showmedians=False)    
    # # for pc in parts['bodies']:
    # #     pc.set_facecolor('#DDA0DD')
    # #     pc.set_edgecolor(colors[0])
    # #     pc.set_alpha(1)
    # # quartile1, medians, quartile3 = np.percentile(iON_iOFF, [25, 50, 75], axis=1)
    # # inds = np.arange(1, len(medians) + 1)
    
    # # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=7)
    # # ax.vlines(inds, quartile1, quartile3, color=colors[0], linestyle='solid', lw=9)
    # # ax.vlines(1, np.amin(iON), np.amax(iON), color=colors[0], linestyle='solid', lw=1)
    # # ax.vlines(2, np.amin(iOFF), np.amax(iOFF), color=colors[0], linestyle='solid', lw=1)

    # # plt.savefig(f_pdir+'/unmapped/'+br+'-integrated_responses-RGECO.png',dpi=300,bbox_inches = 'tight')
    # # plt.show()
    # # plt.close()     

    # #plot and save integrated normal ER flash responses (violin plots)
    # iONer_iOFFer = []
    # iONer_iOFFer.append(iONer)
    # iONer_iOFFer.append(iOFFer)
    # iONer_iOFFer_stack = np.asarray(iONer_iOFFer)
    # plt.figure(figsize=(fs,fs))
    # ax = plt.subplot(111)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.set_xticks([1,2])
    # ax.set_xticklabels(['ON','OFF'], fontsize = 16)
    # plt.ylabel('ER [a.u.]',fontsize = 16)
    # plt.yticks(fontsize=14)
    
    # parts = ax.violinplot(iONer_iOFFer,positions=[1,2], vert=True, showmeans=False,showextrema=False, showmedians=False)    
    # for pc in parts['bodies']:
    #     pc.set_facecolor('#99ff99')
    #     pc.set_edgecolor(colors[1])
    #     pc.set_alpha(1)
    # quartile1, medians, quartile3 = np.percentile(iONer_iOFFer, [25, 50, 75], axis=1)
    # inds = np.arange(1, len(medians) + 1)
    
    # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=7)
    # ax.vlines(inds, quartile1, quartile3, color=colors[1], linestyle='solid', lw=9)
    # ax.vlines(1, np.amin(iONer), np.amax(iONer), color=colors[1], linestyle='solid', lw=1)
    # ax.vlines(2, np.amin(iOFFer), np.amax(iOFFer), color=colors[1], linestyle='solid', lw=1)    
    
    # plt.savefig(f_pdir+'/unmapped/'+br+'-integrated_responses-ER210.png',dpi=300,bbox_inches = 'tight')
    # plt.show()
    # plt.close()
    
    # # #mapped CYT gratings responses 
    # # M1 = np.mean(mapg_CYT,axis=0)
    # # STD1 = np.std(mapg_CYT,axis=0)
    # # STE1 = STD1/np.sqrt(len(M1))
    
    # # plt.figure(figsize=(fs, fs))
    # # ax = plt.subplot(111)  
    # # ax.spines["top"].set_visible(False)
    # # ax.spines["right"].set_visible(False)
    # # ax.get_xaxis().tick_bottom()
    # # ax.get_yaxis().tick_left()
    # # # plt.ylim(yming,ymaxg)
    # # plt.xlim(0,30)
    # # plt.xticks(np.arange(0,31,5), fontsize=14)
    # # # plt.yticks(np.arange(ytming, ytmaxg, ytig), fontsize=14)
    # # plt.yticks(fontsize=14)
    # # plt.xlabel('seconds',fontsize = 14)
    # # plt.ylabel('DF/F',fontsize = 14)
    # # # plt.title(br+' RGECO',fontsize = 16)
    
    # # plt.fill_between(bins,M1+STE1,M1-STE1,color = colors[0],alpha = 0.2)    
    # # plt.axvline(x=5, color=colors[2], linestyle='-.')
    # # plt.axvline(x=15, color=colors[2], linestyle='-.')
    # # plt.plot(bins,M1,linewidth = 1.75, color = colors[0])
    # # plt.savefig(g_pdir+'/mapped_RF/'+br+'-average_responses-RGECO.png',bbox_inches = 'tight',dpi=300)
    # # plt.show()
    # # plt.close()
    
    # # #mapped ER gratings responses 
    # # M2 = np.mean(mapg_ER,axis=0)
    # # STD2 = np.std(mapg_ER,axis=0)
    # # STE2 = STD2/np.sqrt(len(M2))
    
    # # plt.figure(figsize=(fs, fs))
    # # ax = plt.subplot(111)  
    # # ax.spines["top"].set_visible(False)
    # # ax.spines["right"].set_visible(False)
    # # ax.get_xaxis().tick_bottom()
    # # ax.get_yaxis().tick_left()
    # # # plt.ylim(yminerg,ymaxerg)
    # # plt.xlim(0,30)
    # # plt.xticks(np.arange(0,31,5), fontsize=14)
    # # # plt.yticks(np.arange(ytminerg, ytmaxerg, ytierg), fontsize=14)
    # # plt.yticks(fontsize=14)
    # # plt.xlabel('seconds',fontsize = 14)
    # # plt.ylabel('DF/F',fontsize = 14)
    # # # plt.title(br+' ER210',fontsize = 16)
    
    # # plt.axvline(x=5, color=colors[2], linestyle='-.')
    # # plt.axvline(x=15, color=colors[2], linestyle='-.')
    # # plt.fill_between(bins,M2+STE2,M2-STE2,color = colors[1],alpha = alphas[2])
    # # plt.plot(bins,M2,linewidth = 1.75, color = colors[1])
    # # plt.savefig(g_pdir+'/mapped_RF/'+br+'-average_responses-ER210.png',bbox_inches = 'tight',dpi=300)
    # # plt.show()
    # # plt.close()
    
    # # #unmapped CYT gratings responses 
    # # M1 = np.mean(umapg_CYT,axis=0)
    # # STD1 = np.std(umapg_CYT,axis=0)
    # # STE1 = STD1/np.sqrt(len(M1))
    
    # # plt.figure(figsize=(fs, fs))
    # # ax = plt.subplot(111)  
    # # ax.spines["top"].set_visible(False)
    # # ax.spines["right"].set_visible(False)
    # # ax.get_xaxis().tick_bottom()
    # # ax.get_yaxis().tick_left()
    # # # plt.ylim(yming,ymaxg)
    # # plt.xlim(0,30)
    # # plt.xticks(np.arange(0,31,5), fontsize=14)
    # # # plt.yticks(np.arange(ytming, ytmaxg, ytig), fontsize=14)
    # # plt.yticks(fontsize=14)
    # # plt.xlabel('seconds',fontsize = 14)
    # # plt.ylabel('DF/F',fontsize = 14)
    # # # plt.title(br+' RGECO',fontsize = 16)
    
    # # plt.fill_between(bins,M1+STE1,M1-STE1,color = colors[0],alpha = 0.2)    
    # # plt.axvline(x=5, color=colors[2], linestyle='-.')
    # # plt.axvline(x=15, color=colors[2], linestyle='-.')
    # # plt.plot(bins,M1,linewidth = 1.75, color = colors[0])
    # # plt.savefig(g_pdir+'/unmapped_RF/'+br+'-average_responses-RGECO.png',bbox_inches = 'tight',dpi=300)
    # # plt.show()
    # # plt.close()
    
    # # #unmapped ER gratings responses 
    # # M2 = np.mean(umapg_ER,axis=0)
    # # STD2 = np.std(umapg_ER,axis=0)
    # # STE2 = STD2/np.sqrt(len(M2))
    
    # # plt.figure(figsize=(fs, fs))
    # # ax = plt.subplot(111)  
    # # ax.spines["top"].set_visible(False)
    # # ax.spines["right"].set_visible(False)
    # # ax.get_xaxis().tick_bottom()
    # # ax.get_yaxis().tick_left()
    # # # plt.ylim(yminerg,ymaxerg)
    # # plt.xlim(0,30)
    # # plt.xticks(np.arange(0,31,5), fontsize=14)
    # # # plt.yticks(np.arange(ytminerg, ytmaxerg, ytierg), fontsize=14)
    # # plt.yticks(fontsize=14)
    # # plt.xlabel('seconds',fontsize = 14)
    # # plt.ylabel('DF/F',fontsize = 14)
    # # # plt.title(br+' ER210',fontsize = 16)
    
    # # plt.axvline(x=5, color=colors[2], linestyle='-.')
    # # plt.axvline(x=15, color=colors[2], linestyle='-.')
    # # plt.fill_between(bins,M2+STE2,M2-STE2,color = colors[1],alpha = alphas[2])
    # # plt.plot(bins,M2,linewidth = 1.75, color = colors[1])
    # # plt.savefig(g_pdir+'/unmapped_RF/'+br+'-average_responses-ER210.png',bbox_inches = 'tight',dpi=300)
    # # plt.show()
    # # plt.close()
    