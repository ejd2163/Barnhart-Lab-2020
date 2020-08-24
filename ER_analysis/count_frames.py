import numpy
from celltool.utility.path import path
import celltool.utility.datafile as datafile
import matplotlib.pyplot as plt

threshold = 1

"""Set up path"""
parent_dir = path('C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi9/moving_dark_bar/')

"""Import list of directories"""
header,rows = datafile.DataFile(parent_dir/'directories.csv').get_header_and_data()
dir_list = [parent_dir/row[1] for row in rows]

#new_dl for indexing samples to fix
# new_dl = []
# indexes=[13,18]
# for i in indexes: 
#     new_dl.append(dir_list[i])

"""Read stimulus file, count imaging frames, and save output stimulus file for all directories"""
for directory in dir_list[:]: #change list to dir_list or new_dl
    
    print(directory)
    
    """Import original stimulus file to a numpy array; you'll have to change the name of the file for your particular stimulus"""
    stim_file = directory.files('moving*.csv')[0]
    header,rows = datafile.DataFile(stim_file).get_header_and_data()
    R = numpy.asarray(rows)
    
    """Set up the output array (just adds another column to the original stimulus array."""
    output_array = numpy.zeros((R.shape[0],R.shape[1]+1))
    header.extend(['frames'])
    
    """Calculate the change in the voltage signal for each stimulus frame."""
    vs = [0]
    vs.extend(R[1:,-1]-R[:-1,-1]) #allrows,last column minus lastrow, last column
    # R[:-1,-1], print(vs)
    
    """Plot to check, if you want."""
    plt.plot(vs)
    plt.plot(vs,'ko')
    plt.savefig(directory/'plots/frame_timing.png')
    # plt.show()
    plt.close()
    
    
    """And...count imaging frames from the change in voltage signal!"""
    count_on = 0
    F_on = [0]
    count_off = 0
    F_off = [0]
    
    frame_labels = [0]
    
    n = 1
    while n<len(vs)-1:
        if vs[n]>vs[n-1] and vs[n]>vs[n+1] and vs[n] > threshold:
            count_on = count_on+1
            F_on.extend([count_on])
            F_off.extend([count_off])
        elif vs[n]<vs[n-1] and vs[n]<vs[n+1] and vs[n] < threshold*-1:
            count_off = count_off-1
            F_off.extend([count_off])
            F_on.extend([F_on])
        else:
            F_on.extend([count_on])
            F_off.extend([count_off])
        frame_labels.extend([count_on*(count_on+count_off)])
        n=n+1
    frame_labels.extend([0])
    # print(frame_labels.index(max(frame_labels)))
    # print(F_on)
    # print(F_off)
    
    plt.plot(frame_labels)
    plt.savefig(directory/'plots/frame_numbers.png')
    plt.show()
    plt.close()
    
    print(count_on)
    print(count_off)
    
    output_array[:,:R.shape[1]] = R
    output_array[:,-1] = frame_labels
    
    OAS = output_array[output_array[:,-1].argsort()]
    i1 = numpy.searchsorted(OAS[:,-1],1)
    
    datafile.write_data_file([header]+list(OAS[i1:,:]),directory/stim_file.namebase+'-frames.csv')


