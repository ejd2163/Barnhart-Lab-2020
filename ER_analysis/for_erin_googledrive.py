import os
import shutil
from celltool.utility.path import path
import celltool.utility.datafile as datafile

stim='10s_flashes'

"""Set up path"""
directory = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Tm3/medulla/'+stim
parent_dir = path(directory)
directories = os.listdir(parent_dir)

t_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Tm3/medulla/google_drive/'
target_dir = path(t_dir)
dirs = os.listdir(target_dir)

header,rows = datafile.DataFile(parent_dir/'directories.csv').get_header_and_data()

path_list = [parent_dir/row[1] for row in rows]

##0) to rename files (optional)
# for row in directories[1:19]: 
    
#     prev=os.path.join(directory+row,'mask-M3-M4.tif')
#     new = os.path.join(directory+row,'mask-M2-M5.tif')
#     os.rename(prev,new)

# 1) CREATE DIRECTORIES FOR GOOGLE DRIVE FOLDER 
# i=1
# for row in rows[:]:

    
#     targetname=row[1][2:13]+'-z'+str(i)
#     targetfile=os.path.join(t_dir,targetname)
#     os.mkdir(targetfile)
#     os.mkdir(os.path.join(t_dir+targetname,'measurements'))
#     os.mkdir(os.path.join(t_dir+targetname,'stim_files'))
#     i=i+1
   
# # #2) COPY STIM FILES TO GOOGLE DRIVE FOLDER  
# for file,row in zip(dirs[:],path_list[:]): 
    
#     parent_stimfile = row.files('full*.csv')[0]
#     target_stimfile = t_dir+file+'/stim_files/'
#     shutil.copy(parent_stimfile,target_stimfile)
#     parent_stimfile = row.files('full*.csv')[1]
#     target_stimfile = t_dir+file+'/stim_files/'
#     shutil.copy(parent_stimfile,target_stimfile)
        
#     parent_RGECO = os.path.join(row+'/measurements/',stim+'-raw_responses-RGECO.csv')
#     raw_RGECO = os.path.join(t_dir+file+'/measurements/',stim+'-raw_responses-RGECO.csv')
#     shutil.copyfile(parent_RGECO,raw_RGECO)
#     parent_ER210 = os.path.join(row+'/measurements/',stim+'-raw_responses-ER210.csv')
#     raw_ER210 = os.path.join(t_dir+file+'/measurements/',stim+'-raw_responses-ER210.csv')
#     shutil.copyfile(parent_ER210,raw_ER210)
    


