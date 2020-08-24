# import relevant modules
import numpy as np
import celltool.utility.datafile as datafile

#relevant compartment regions
brain_regions = ['M1','M5','M9-M10'] 

#moving light bars directory name
mlb_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/moving_light_bars/'
mlb_mdir= mlb_dir+'/measurements/2s/'
map_mdir=mlb_mdir+'/mappable/'
unmap_mdir=mlb_mdir+'/unmappable/'

#input flashes directory name
flashes_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi1/2s_flashes/'
f_mdir = flashes_dir + '/measurements/'
f_pdir = flashes_dir +'/plots/'
 
# =============================================================================
# LABELING ROIS BELOW: 
# from 'mapped' plots, list ROI numbers designated in respective plot names
    #that should be considered 'unmapped RF' responses in 'to_unmap' strings
# from 'unmapped' plots, list ROI numbers designated in respective plot names
    #that should be considered having 'mapped RF' responses in 'to_map' strings
# =============================================================================
#M1
to_unmap_M1 = [0,1,53,66,67,68,69,70,71,72,73,77,78,86,93,133,\
	152,160]
to_map_M1 = [35,36,39,43,44,63,87]
#M5
to_unmap_M5=[0,1,2,3,4,5,6,7,17,27,30,73,96,97,105,106,107,108,\
	109,110,111,112,113,116,117,120,157,169,172,179,180,181,\
	182,196,197,199,215,216]
to_map_M5=[56,57,86,87]
#M9-M10
to_unmap_axons=[0,1,2,3,4,5,24,28,95,107,112,113,114,115,116,\
	117,118,119,120,121,122,123,124,125,126,127,128,129,\
	137,138,143,144,145,146,147,148,240,254,255,262,263,264]
to_map_axons=[45,47,51,60,64,69,118,119,120,121,122,123,206,209]

to_unmap_indexes=[to_unmap_M1,to_unmap_M5,to_unmap_axons]
to_map_indexes=[to_map_M1,to_map_M5,to_map_axons]

for br,tounmap,tomap in zip(brain_regions[:],to_unmap_indexes[:],to_map_indexes[:]):
    print('BRAIN REGION '+br)
    
    #______SCREEN RF CENTERS____________    
    
    #import MLB data w/ mapped and unmapped RF centers and separate headers and rows from each dataset
    header,rows = datafile.DataFile(map_mdir+'RF_centers-'+br+'.csv').get_header_and_data() #mapped
    mapRF = np.asarray(rows)    
    header,rows = datafile.DataFile(unmap_mdir+'RF_centers-'+br+'.csv').get_header_and_data() #unmapped
    unmapRF = np.asarray(rows)
    print('RF centers total: '+str(len(mapRF)+len(unmapRF)))    

    #import flash ER and RGECO data w/ mapped RF centers and separate headers and rows from each dataset
    headerf,rows = datafile.DataFile(f_mdir+'/mapped/average_responses-'+br+'-ER210.csv').get_header_and_data()
    fmapER = np.asarray(rows)
    headerf,rows = datafile.DataFile(f_mdir+'/mapped/average_responses-'+br+'-RGECO.csv').get_header_and_data()
    fmapCYT = np.asarray(rows)
    
    #import flash ER and RGECO data w/ unmapped RF centers and separate headers and rows from each dataset
    headerf,rows = datafile.DataFile(f_mdir+'/unmapped/average_responses-'+br+'-ER210.csv').get_header_and_data()
    funmapER = np.asarray(rows)
    headerf,rows = datafile.DataFile(f_mdir+'/unmapped/average_responses-'+br+'-RGECO.csv').get_header_and_data()
    funmapCYT = np.asarray(rows)
    print('flashes total: '+str(len(fmapCYT)/2+len(funmapCYT)/2))  
    
    #extract unmappable RFs from mappable RF_centers array
    sub_unmap=[]
    newmap=[]   
    sub_funmapCYT=[]
    newfmapCYT=[] 
    sub_funmapER=[]
    newfmapER=[] 
    m=0        
    b=0
    e=0
    if len(tounmap)==0: 
        [newmap.append(mapRF[n,:]) for n in range(len(mapRF))]  
        [newfmapCYT.append(fmapCYT[n,:]) for n in range(len(fmapCYT))]
        [newfmapER.append(fmapER[n,:]) for n in range(len(fmapER))]
    else:
        while b<len(mapRF) and m<len(tounmap): #b<len(mapRF)
            if tounmap[m]==b: #if 68==n: 
                sub_unmap.append(mapRF[b,:])
                sub_funmapCYT.append(fmapCYT[e,:])
                sub_funmapCYT.append(fmapCYT[e+1,:])
                sub_funmapER.append(fmapER[e,:])
                sub_funmapER.append(fmapER[e+1,:])
                m=m+1
            else:
                newmap.append(mapRF[b,:])  
                newfmapCYT.append(fmapCYT[e,:])
                newfmapCYT.append(fmapCYT[e+1,:])
                newfmapER.append(fmapER[e,:])
                newfmapER.append(fmapER[e+1,:])
            b=b+1
            e=e+2

    if len(tounmap)>0 and tounmap[-1]<len(mapRF): 
        a = int(tounmap[-1])           
        [newmap.append(mapRF[a+i+1,:]) for i in range(len(mapRF)-a-1)]   
        a = 2*int(tounmap[-1])
        [newfmapCYT.append(fmapCYT[a+i+2,:]) for i in range(len(fmapCYT)-a-2)]
        [newfmapER.append(fmapER[a+i+2,:]) for i in range(len(fmapER)-a-2)]

    print('# of initial screened RF: '+str(len(newmap)))
    print('# of initial screened flash: '+str(len(newfmapCYT)/2)) 
    
    #extract mappable RFs from unmappable RF_centers array
    sub_map=[]
    newunmap=[]   
    sub_fmapCYT=[]
    newfunmapCYT=[] 
    sub_fmapER=[]
    newfunmapER=[] 
    if len(tomap)==0: 
        [newunmap.append(unmapRF[n,:]) for n in range(len(unmapRF))] 
        [newfunmapCYT.append(funmapCYT[n,:]) for n in range(len(funmapCYT))]
        [newfunmapER.append(funmapER[n,:]) for n in range(len(funmapER))]
    else:
        m=0        
        n=0
        f=0
        while n<int(tomap[-1])+1 and m<len(tomap):
            if n==tomap[m]:
                sub_map.append(unmapRF[n,:])
                sub_fmapCYT.append(funmapCYT[f,:])
                sub_fmapCYT.append(funmapCYT[f+1,:])
                sub_fmapER.append(funmapER[f,:])
                sub_fmapER.append(funmapER[f+1,:])
                m=m+1
            else:
                newunmap.append(unmapRF[n,:])  
                newfunmapCYT.append(funmapCYT[f,:])
                newfunmapCYT.append(funmapCYT[f+1,:])
                newfunmapER.append(funmapER[f,:])
                newfunmapER.append(funmapER[f+1,:])
            n=n+1 
            f=f+2
    if len(tomap)>0 and tomap[-1]<len(unmapRF): 
        a = int(tomap[-1])           
        [newunmap.append(unmapRF[a+i+1,:]) for i in range(len(unmapRF)-a-1)]
        a = 2*int(tomap[-1])
        [newfunmapCYT.append(funmapCYT[a+i+2,:]) for i in range(len(funmapCYT)-a-2)]
        [newfunmapER.append(funmapER[a+i+2,:]) for i in range(len(funmapER)-a-2)]
        
    #add mapped RFs to updated mapped RFs array
    sub_map=np.asarray(sub_map)
    sub_fmapCYT=np.asarray(sub_fmapCYT)
    sub_fmapER=np.asarray(sub_fmapER)
    [newmap.append(sub_map[a,:]) for a in range(len(sub_map))]
    [newfmapCYT.append(sub_fmapCYT[a,:]) for a in range(len(sub_fmapCYT))]
    [newfmapER.append(sub_fmapER[a,:]) for a in range(len(sub_fmapER))] 

    print('# of total mapped RF: '+str(len(newmap)))
    print('# of total mapped flash: '+str(len(newfmapCYT)/2)) 
    
    #add unmapped RFs to updated unmapped RFs array
    sub_unmap=np.asarray(sub_unmap)
    sub_funmapCYT=np.asarray(sub_funmapCYT)
    sub_funmapER=np.asarray(sub_funmapER)
    [newunmap.append(sub_unmap[a,:]) for a in range(len(sub_unmap))]  
    [newfunmapCYT.append(sub_funmapCYT[a,:]) for a in range(len(sub_funmapCYT))]
    [newfunmapER.append(sub_funmapER[a,:]) for a in range(len(sub_funmapER))]    

    #save new mapped and unmapped RFs arrays as .csv files
    datafile.write_data_file([header]+newmap,map_mdir+'/newRF_centers-'+br+'.csv') #MLB RF centers .csv
    datafile.write_data_file([header]+newunmap,unmap_mdir+'/newRF_centers-'+br+'.csv') #MLB RF centers .csv
    datafile.write_data_file([headerf]+newfmapER,f_mdir+'/mapped/newavg_flash-'+br+'-ER210.csv') #ER210 flash responses .csv (mapped RFs)
    datafile.write_data_file([headerf]+newfmapCYT,f_mdir+'/mapped/newavg_flash-'+br+'-RGECO.csv') #RGECO flash responses .csv (mapped RFs)
    datafile.write_data_file([headerf]+newfunmapER,f_mdir+'/unmapped/newavg_flash-'+br+'-ER210.csv') #ER210 flash responses .csv (unmapped RFs)
    datafile.write_data_file([headerf]+newfunmapCYT,f_mdir+'/unmapped/newavg_flash-'+br+'-RGECO.csv')

