import numpy
import numpy as np
from pylab import *
import csv
import matplotlib.pyplot as plt
import pandas as pd 
#from scipy import signal

#rotate clockwise, right side of flux plane is distal


calibration_pix = 0.091
calibration= calibration_pix


normal = numpy.asarray([0,-1])

# header,rows = datafile.DataFile('MitosTracked.csv').get_header_and_data()
# centroids = numpy.asarray(rows)
# centroids = centroids[:,2:4]
# velocities= numpy.asarray(rows)
# velocities = velocities[:,5:6]
threshold= 0.1
time_interval= 1
data_dict = {} #dictionary--> pairwise, key is mitchondrial track (1,2,3 ect.) Value is a two-pule list of x values and list of y values
list_of_rows = [] #list of rows
movingmitos=[0,0]

with open('MitoTracked.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ', quotechar='|') #opens and reads csv file 
	firstLine = True # Boolean which ignores the header row --> x, y, area header ect. 
	for row in reader:
		if firstLine:
			firstLine = False
			continue

		data = row[0].split(",") #breaks down the componets of the row
		list_of_rows.append(data) #Adds all the rows with separated column values together 


#Create a function to get list of tracks (ex. [1,3,5,7])
def get_list_of_tracks():
	list_of_tracks = []
	for row in list_of_rows:
		if row[0] not in list_of_tracks:
			list_of_tracks.append(row[0])  
	return list_of_tracks 

#Create a function to pass a value through the function --> groups together tracks  
def get_data_tuple_for_track(track):
	X = []
	Y = []
	for row in list_of_rows:
		if row[0] == track:
			X.append(float(row[2])) # Adds to list of x values for each track, use float to convert strings to float
			Y.append(float(row[3])) # Adds to list of y values for each track 
#			Vel.append(float(row[5]))
	return (X,Y)

#Create a function to define the tracks paired with the subsequent rows
def build_data_dictionary():
	tracks = get_list_of_tracks()
	for track in tracks:
		data_dict[track] = get_data_tuple_for_track(track)

# function to determine track directionality (if track is retrograde or antrograde)
def determine_direction(first_Y, last_Y):
	return first_Y < last_Y


velocities = []
speedsBlue = []
speedsGold = []
propertiesBlue = []
propertiesGold = []
Bluevelocities= []
Goldvelocities= []


def build_centroids():
	#loop through tracks on data dictionary. At each track turn X, Y into an array. Call cal_speed. 
		for track in sorted(data_dict.keys()):
			X_data = data_dict[track][0]
			Y_data = data_dict[track][1]
			centroids = []
			

			for i in range(len(X_data)):
				centroids.append([X_data[i], Y_data[i]])
			

#				X_smooth = signal.savgol_filter(centroids[:,0],3,2)
#				Y_smooth = signal.savgol_filter(centroids[:,1],3,2)
#			print(centroid2)
#			print(centroids)
			if (determine_direction(Y_data[0],Y_data[-1])):
				Bluevelocities.append(calculate_velocities(numpy.array(centroids), time_interval, calibration))
			else:
				Goldvelocities.append(calculate_velocities(numpy.array(centroids), time_interval, calibration))	
#			velocities.append(calculate_velocities(numpy.array(centroids), time_interval, calibration))
			
			if (determine_direction(Y_data[0],Y_data[-1])):
				speedsBlue.append(calculate_speed(numpy.array(centroids), time_interval, calibration))
			else:
				speedsGold.append(calculate_speed(numpy.array(centroids), time_interval, calibration))		
			
				
		return numpy.array(centroids)
#			axs[0].plot(centroid2[:,0],centroid2[:,1],'ko')
#			axs[0].set_xlim(75,175)
#			axs[0].set_ylim(150,250)
#			axs[1].plot(X_smooth,Y_smooth,'r')			
			
def build_velocity_plot():
	for track_velocity in velocities:
		y_values = track_velocity.tolist();
		x_values = list(range(0, len(y_values)))
		axs[2].set_xlabel('Inital to Final Appearance')
		axs[2].set_ylabel('Velocity')
		if (determine_direction(y_values[0], y_values[-1])):
			axs[2].plot(x_values, y_values, 'b')
		else:
			axs[2].plot(x_values, y_values, 'y')

def calculate_speed(centroids, time_interval, calibration):
	displacement_vectors = centroids[1:] - centroids[:-1]
	#print(numpy.sqrt(numpy.sum(displacement_vectors**2,axis = 1))/time_interval*calibration)
	#print(type(numpy.sqrt(numpy.sum(displacement_vectors**2,axis = 1))/time_interval*calibration))
	return numpy.sqrt(numpy.sum(displacement_vectors**2,axis = 1))/time_interval*calibration

def build_speed_plot():
	means_Blue=[]
	means_Gold=[]
	first_row= 0
	for track_speed in speedsBlue:
		means_Blue.append(numpy.mean(track_speed))

	for track_Gold_speed in speedsGold:
		means_Gold.append(numpy.mean(track_Gold_speed))

	#print(means_Blue)
	#print('Line')
	#print(means_Gold)
	np.savetxt("Blue.csv", means_Blue, delimiter=",", fmt='%s')
	np.savetxt("Gold.csv", means_Gold, delimiter=",", fmt='%s')

		# x_values = list(range(0, len(y_values)))
		# axs[4].set_xlabel('Inital to Final Appearance')
		# axs[4].set_ylabel('Speed')
		# if (determine_direction(y_values[0], y_values[-1])):
		# 	axs[4].plot(x_values, y_values, 'b')
		# else:
		# 	axs[4].plot(x_values, y_values, 'y')
		# axs[4].plot(x_values, y_values)



def calculate_velocities(centroids, time_interval, calibration, normal = numpy.asarray([1,0])):
	displacement_vectors = centroids[1:] - centroids[:-1]
	distances = numpy.sqrt((displacement_vectors**2).sum(axis=1))
	dot_products = (normal * displacement_vectors).sum(axis=1)
#	print(numpy.sign(dot_products)*distances*calibration/time_interval)
	return numpy.sign(dot_products)*distances*calibration/time_interval



	#print(means_Blue)
	#print('Line')
	#print(means_Gold)
	np.savetxt("Blue.csv", means_Blue, delimiter=",", fmt='%s')
	np.savetxt("Gold.csv", means_Gold, delimiter=",", fmt='%s')




def build_properties():
	for track_velocity in Bluevelocities:
		if len(track_velocity)!=0:
			propertiesBlue.append(measure_Bluemovement_properties(track_velocity))
			
	for track_velocity in Goldvelocities:
		if len(track_velocity)!=0:
			propertiesGold.append(measure_Goldmovement_properties(track_velocity))

			

# def build_properties():
# 	for track_velocity in velocities:
# 		if len(track_velocity)!=0:
# 			properties.append(measure_movement_properties(track_velocity))

def measure_Bluemovement_properties(Bluevelocities):
	#measure the frequency (per second) with which the mito stops moving or changes direction, as well as the fraction of time spent in motion, and the mean velocity (while in motion).  The threshold parameter is the speed below which the mito is considered stopped; time_interval is the time between frames, in seconds.
	#ED: What isvelocities?  threshold? 

	#print("=========================")
	stopped = 0
	reversed = 0
	moving = []
	n=0
	while n<len(Bluevelocities)-1:
		# print(n)
		# print(velocities)
		# print(velocities[n])
		# print(numpy.abs(velocities[n]))
		if numpy.abs(Bluevelocities[n]) > threshold:
			moving.append(Bluevelocities[n])
		if numpy.abs(Bluevelocities[n]) > threshold and len(moving)>1:
			if numpy.sign(Bluevelocities[n])!=numpy.sign(moving[-2]):
				reversed = reversed + 1
		if numpy.abs(Bluevelocities[n]) > threshold and numpy.abs(Bluevelocities[n+1]) < threshold:
			stopped = stopped + 1
		n=n+1
	np.savetxt("BlueProperties.csv", propertiesBlue, delimiter=",", fmt='%s')

	# print(stopped)
	# print(reversed)
	# print(moving)
	# print("=============$%$%$%$%============")
	# print(float(len(velocities)*time_interval))
	# print(len(velocities))
	# print(float(stopped)/float(len(velocities)*time_interval))
	# print(float(reversed)/float(len(velocities)*time_interval))
	# print(float(len(moving))/float(len(velocities)))
	# print(float(numpy.mean(numpy.abs(moving))))
	# print(float(numpy.mean(moving)))
	return [float(stopped)/float(len(Bluevelocities)*time_interval), float(reversed)/float(len(Bluevelocities)*time_interval), float(len(moving))/float(len(Bluevelocities)), float(numpy.mean(numpy.abs(moving))),float(numpy.mean(moving))]
	np.savetxt("BlueProerties.csv", propertiesBlue, delimiter=",", fmt='%s')

def measure_Goldmovement_properties(Goldvelocities):
	#measure the frequency (per second) with which the mito stops moving or changes direction, as well as the fraction of time spent in motion, and the mean velocity (while in motion).  The threshold parameter is the speed below which the mito is considered stopped; time_interval is the time between frames, in seconds.
	#ED: What isvelocities?  threshold? 

	#print("=========================")
	stopped = 0
	reversed = 0
	moving = []
	n=0
	while n<len(Goldvelocities)-1:
		# print(n)
		# print(velocities)
		# print(velocities[n])
		# print(numpy.abs(velocities[n]))
		if numpy.abs(Goldvelocities[n]) > threshold:
			moving.append(Goldvelocities[n])
		if numpy.abs(Goldvelocities[n]) > threshold and len(moving)>1:
			if numpy.sign(Goldvelocities[n])!=numpy.sign(moving[-2]):
				reversed = reversed + 1
		if numpy.abs(Goldvelocities[n]) > threshold and numpy.abs(Goldvelocities[n+1]) < threshold:
			stopped = stopped + 1
		n=n+1
	# print(stopped)
	# print(reversed)
	# print(moving)
	# print("=============$%$%$%$%============")
	# print(float(len(velocities)*time_interval))
	# print(len(velocities))
	# print(float(stopped)/float(len(velocities)*time_interval))
	# print(float(reversed)/float(len(velocities)*time_interval))
	# print(float(len(moving))/float(len(velocities)))
	# print(float(numpy.mean(numpy.abs(moving))))
	# print(float(numpy.mean(moving)))
	np.savetxt("GoldProperties.csv", propertiesGold, delimiter=",", fmt='%s')
	return [float(stopped)/float(len(Goldvelocities)*time_interval), float(reversed)/float(len(Goldvelocities)*time_interval), float(len(moving))/float(len(Goldvelocities)), float(numpy.mean(numpy.abs(moving))),float(numpy.mean(moving))]
	
# def build_propertiessummary_plot():
# 	summaryproperties_Blue=[]
# 	summaryproperties_Gold=[]
# 	first_row= 0
# 	for track_properties in propertiesBlue:
# 		summaryproperties_Blue.append(numpy.mean(track_speed))

# 	for track_Gold_properties in propertiesGold:
# 		summaryproperties_Gold.append(numpy.mean(track_Gold_speed))

# 	#print(means_Blue)
# 	#print('Line')
	#print(means_Gold)
	





#print(calculate_speed(centroids, 1, calibration))
#print(calculate_velocities(centroids, 1, calibration, normal))
#print(measure_movement_properties(velocities, threshold, time_interval))


fig, axs = plt.subplots(5)
fig.tight_layout(pad=3.0)

build_data_dictionary()
build_centroids()
# print('pppppppppppppppppp')
# print(Bluevelocities)
# print('==============')
# print(Goldvelocities)
# print('oooooooooooooo')
build_properties()
# print('AAAAAA')
# print(len(velocities))
# print('BBBB')
# print(velocities)
# print('ccccc')
#calculate_velocities()
print('==============')
print(propertiesBlue)
print('pppppppppppppppppp')
print(propertiesGold)
# build_measure_movement_properties(velocities, threshold, time_interval)


build_speed_plot()


plt.show();





