import numpy
import celltool.utility.datafile as datafile
import matplotlib.pyplot as plt 

def smooth(x,window_len=5,window='hanning'): #window_len = 5
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
t = 0.1

ymin = -0.3
ymax = 3.0
ytmin = 0
ytmax = 3.1
yti = 0.3

T = numpy.arange(0,5,t)

#convert time series into screen position series
p1=T*4.02-10.03 #screen x (left -> right)
p2=T*-4.02+10.03 #screen x (right -> left)
p3=T*4.02-10.03 #screen y (bottom -> top)
p4=T*-4.02+10.03 #screen y (top -> bottom)

P = [p1,p2,p3,p4] #screen positions series for ea/ epoch

threshold = 0.10 #threshold for RGECO responses

header_out = ['sample','br','ROI','cell_x','cell_y','screen_x','screen_y','amplitude']

ENUM=[0,1,2,3] #epoch number
colors = ['#8F02C5','#02C54F','#0F37FF','#F8A51B'] #purple, green, blue, orange
brain_regions = ['M1','M5','M9-M10']

parent_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/'
MLB_dir = parent_dir+'/moving_light_bars/'
flash_mdir = parent_dir+'/10s_flashes/measurements/'
# gratings_mdir = parent_dir +'/gratings/measurements/'

out_mdir = MLB_dir+'/measurements/2s'
map_mdir = out_mdir+'/mappable/'
unmap_mdir = out_mdir+'/unmappable/'

plot_dir = MLB_dir+'/plots/2s'
map_pdir = plot_dir+'/mappable/'
unmap_pdir = plot_dir+'/unmappable/'

for br in brain_regions[:]:
    
    print('BRAIN REGION '+str(br))
    
    #RGECO moving light bars data
    OUT=[] #for ROIS w/ mappable RFs
    OUT2 =[] #for ROIs w/ unmappable RFs
    header,rows = datafile.DataFile(out_mdir+'/average_responses-'+br+'.csv').get_header_and_data()
    R = numpy.asarray(rows)
    print('mlb count: '+str(len(R)/4))
    
    #RGECO flash responses data
    norm_cytf=[] #empty list for RFs pointing at the visual screen
    abn_cytf=[] #empty list for RFs not pointing at the screen or RFs having weak responses
    flash_header,rows = datafile.DataFile(flash_mdir+'/average_responses-'+br+'-RGECO.csv').get_header_and_data()
    flash_cyt = numpy.asarray(rows)
    print('flash count: '+str(len(flash_cyt)/2))
    
    #ER210 flash responses data
    norm_ERf=[] #empty list for ER responses correlated w/ RFs pointing at the visual screen
    abn_ERf=[] #empty list for ER responses correlated w/ RFs not pointing at screen or RFs having weak responses
    flash_header,rows = datafile.DataFile(flash_mdir+'/average_responses-'+br+'-ER210.csv').get_header_and_data()
    flash_ER = numpy.asarray(rows)   
    
    # print('Analyzing RGECO channel...')
    f = 0
    g = 0
    n = 0
    count_map=0
    count_unmap=0
    while n<len(R) and f<len(flash_cyt)-1:  #and g<len(g_cyt)
    #while n==8:
        # print(R[n,2]) #ROI number
        o = list(R[n,:5]) #sample, br, ROI, cell_x, cell_y
        o2 = list(R[n,:5])
        rROI = numpy.asarray(R[n:n+4,6:],dtype='float') #time series of all 4 epochs for a single ROI
        plt.figure(figsize=(fs, fs))
        ax = plt.subplot(111)  
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # plt.ylim(ymin,ymax)
        plt.xlim(-7,7) #width of the screen 
        plt.xticks(numpy.arange(-6,6.1,2), fontsize=12)
        # plt.yticks(numpy.arange(ytmin, ytmax, yti), fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('screen position',fontsize = 14)
        plt.ylabel('DF/F',fontsize = 14)
        
        rmax=[] #empty list for max DF/F values (y)
        pmax=[] #empty list for position value (x) where max DF/F value (y) is
        for p,c,epoch_num in zip(P,colors,ENUM):
            avi = smooth(rROI[epoch_num,:]) #smooth ID signal
            avi = avi[2:-2] #smoothing adds 2 additional indexes on either sides of array; must extract relevant info instead
            ax.plot(p,avi,color = c) #plot (screen position series, ROI responses) for ea/ epoch
            ax.plot(p,rROI[epoch_num,:],color=c,alpha=0.2)
            max_index=numpy.argmax(avi) #get the index of the max DF/F value in that epoch

            ax.axvline(p[max_index],color=c, linewidth=0.5) #place vertical lines at that index
            rmax.append(numpy.max(avi))
            pmax.append(p[max_index])
 
        #thresholds
        a=0
        rmax_std=[]
        for i,epoch_num in zip(rmax,ENUM): 
            avi = smooth(rROI[epoch_num,:]) #smooth ID signal
            avi_min = min(avi)
            if i>= 3*numpy.std(avi)+avi_min: # 3 stdevs above the min
                a=True
            else: 
                a=False
            rmax_std.append(a)
        
        #check if each position value is w/in the screen limits
        a=0
        pmax_st=[]
        for i in pmax: 
            if -7<i and i<7: 
                a=True #epoch i is w/in screen limits
            else: 
                a=False #epoch i is not w/in screen limits 
            pmax_st.append(a)
        
        #check if the width between epochs is reasonably narrow (whether peaks of horiz./vert. pair of epochs are next to each other)
        pmax_width=0    
        if abs(pmax[0]-pmax[1])<6 and abs(pmax[2]-pmax[3])<6: 
            pmax_width=True 
        else: 
            pmax_width=False
            
        #to be considered having mapped RFs, sample must fulfill these requirements: 
            #1) mean of max of responses from all 4 epochs must reach threshold
            #2) there must be 4 max DFF in rmax, one from each epoch
            #3) each position from the screen is within screen limits [-7,7]
            #4) width between max values of pair of epochs (horiz. pair/vert. pair) is narrow

        if numpy.mean(rmax_std)==1 and len(rmax)==4 and numpy.mean(pmax_st)==1 and pmax_width==1: #and rmax_noise==1: 
            o.extend([numpy.mean(pmax[:2]),numpy.mean(pmax[2:]),numpy.mean(rmax)]) #screen x, screen y, amplitude (average from all 4 epochs)
            OUT.append(o)
            #append responses to lists for mapped RF responses
            norm_cytf.append(flash_cyt[f,:]) #dark flash RGECO responses
            norm_cytf.append(flash_cyt[f+1,:]) #light flash RGECO responses
            norm_ERf.append(flash_ER[f,:]) #dark flash ER responses
            norm_ERf.append(flash_ER[f+1,:]) #light flash ER responses
            plt.savefig(map_pdir+br+'/'+str(count_map).zfill(4)+'-'+str(R[n,0])+'-'+br+'-ROI_'+str(R[n,2])+'.png',dpi=300,bbox_inches = 'tight')
            count_map=count_map+1
        else: 
            o2.extend([numpy.mean(pmax[:2]),numpy.mean(pmax[2:]),numpy.mean(rmax)]) #screen x, screen y, amplitude (average from all 4 epochs)
            OUT2.append(o2)
            #append responses to lists for unmapped responses
            abn_cytf.append(flash_cyt[f,:])
            abn_cytf.append(flash_cyt[f+1,:])
            abn_ERf.append(flash_ER[f,:])
            abn_ERf.append(flash_ER[f+1,:])
            plt.savefig(unmap_pdir+br+'/'+str(count_unmap).zfill(4)+'-'+str(R[n,0])+'-'+br+'-ROI_'+str(R[n,2])+'.png',dpi=300,bbox_inches = 'tight')
            count_unmap=count_unmap+1


        # plt.show()
        plt.close()
        
        f=f+2
        g=g+1
        n=n+4
    
    #Write and save .csv files 
    datafile.write_data_file([header_out]+OUT,map_mdir+'/RF_centers-'+br+'.csv') #MLB RF centers .csv
    datafile.write_data_file([flash_header]+norm_cytf,flash_mdir+'/mapped/average_responses-'+br+'-RGECO.csv') #flash responses .csv RGECO
    datafile.write_data_file([flash_header]+norm_ERf,flash_mdir+'/mapped/average_responses-'+br+'-ER210.csv') #flash responses .csv ER210
    
    #Write up .csv files for unmappable RF centers
    datafile.write_data_file([header_out]+OUT2,unmap_mdir+'/RF_centers-'+br+'.csv') #MLB RF centers .csv
    datafile.write_data_file([flash_header]+abn_cytf,flash_mdir+'/unmapped/average_responses-'+br+'-RGECO.csv') #flash responses .csv RGECO
    datafile.write_data_file([flash_header]+abn_ERf,flash_mdir+'/unmapped/average_responses-'+br+'-ER210.csv') #flash responses .csv ER210

    #lower threshold RFs 
    #go through ROIs to toss out from mappable 
    #plot both mappable and unmappable RFs ROIs gratings and flashes 
            #make folders for respective categories 
    #PCA for ROIs w/ mappable RFs 
    