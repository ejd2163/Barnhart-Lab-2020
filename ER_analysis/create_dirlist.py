import os
from celltool.utility.path import path
import celltool.utility.datafile as datafile

stim='/moving_dark_bar/'

"""Copy filenames from raw image files to analysis folders"""
# directory = 'D:/Image_files/Mi9_strongER210+RGECO/'+stim
# parent_dir = path(directory)

# dirs = sorted(os.listdir(parent_dir))

t_dir = 'C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi9_strong/'+stim
target_dir = path(t_dir)

#1) CREATE FILES FROM IMAGE_FILES TO NEW_ER/SCREEN FOLDERS
# for file in dirs[:]: 
    
#     targetname=file
#     os.mkdir(os.path.join(t_dir,targetname))
    
# # """2) Create directories.csv file"""
target_list = sorted(os.listdir(target_dir))
header = ['fly','directory']
dir_list = []

for file in target_list: 
    i=1
    if 'fly' in file:
        
        
        i = file.find('fly') + 3
        fly_no = file[i]
                
        dir_list.append([fly_no,file])

datafile.write_data_file([header]+dir_list,target_dir/'directories.csv')
                        