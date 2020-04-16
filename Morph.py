import numpy
from pylab import *
import csv
import matplotlib.pyplot as plt


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

with open('FinalResultsMovement.csv', newline='') as csvfile:
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
speeds = []
def build_centroids():
	#loop through tracks on data dictionary. At each track turn X, Y into an array. Call cal_speed. 
		for track in sorted(data_dict.keys()):
			X_data = data_dict[track][0]
			Y_data = data_dict[track][1]
			centroids = []
			for i in range(len(X_data)):
				centroids.append([X_data[i], Y_data[i]])
			velocities.append(calculate_velocities(numpy.array(centroids), time_interval, calibration))
			speeds.append(calculate_speed(numpy.array(centroids), time_interval, calibration))

def build_velocity_plot():
	for track_velocity in velocities:
		y_values = track_velocity.tolist();
		x_values = list(range(0, len(y_values)))
		axs[0].set_xlabel('Inital to Final Appearance')
		axs[0].set_ylabel('Velocity')
		if (determine_direction(y_values[0], y_values[-1])):
			axs[0].plot(x_values, y_values, 'b')
		else:
			axs[0].plot(x_values, y_values, 'y')


def calculate_speed(centroids, time_interval, calibration):
	displacement_vectors = centroids[1:] - centroids[:-1]
	print(numpy.sqrt(numpy.sum(displacement_vectors**2,axis = 1))/time_interval*calibration)
	return numpy.sqrt(numpy.sum(displacement_vectors**2,axis = 1))/time_interval*calibration

def build_speed_plot():
	for track_speed in speeds:
		y_values = track_speed.tolist();
		x_values = list(range(0, len(y_values)))
		axs[1].set_xlabel('Inital to Final Appearance')
		axs[1].set_ylabel('Speed')
		if (determine_direction(y_values[0], y_values[-1])):
			axs[1].plot(x_values, y_values, 'b')
		else:
			axs[1].plot(x_values, y_values, 'y')
		#axs[1].plot(x_values, y_values)



def calculate_velocities(centroids, time_interval, calibration, normal = numpy.asarray([1,0])):
	displacement_vectors = centroids[1:] - centroids[:-1]
	distances = numpy.sqrt((displacement_vectors**2).sum(axis=1))
	dot_products = (normal * displacement_vectors).sum(axis=1)
	print(numpy.sign(dot_products)*distances*calibration/time_interval)
	return numpy.sign(dot_products)*distances*calibration/time_interval

def measure_movement_properties(velocities, threshold, time_interval):
	#measure the frequency (per second) with which the mito stops moving or changes direction, as well as the fraction of time spent in motion, and the mean velocity (while in motion).  The threshold parameter is the speed below which the mito is considered stopped; time_interval is the time between frames, in seconds.
	#ED: What isvelocities?  threshold? 

	
	stopped = 0
	reversed = 0
	moving = []
	n=0
	while n<len(velocities)-1:
		if numpy.abs(velocities[n]) > threshold:
			moving.append(velocities[n])
		if numpy.abs(velocities[n]) > threshold and len(moving)>1:
			if numpy.sign(velocities[n])!=numpy.sign(moving[-2]):
				reversed = reversed + 1
		if numpy.abs(velocities[n]) > threshold and numpy.abs(velocities[n+1]) < threshold:
			stopped = stopped + 1
		n=n+1
	return [stopped/float(len(velocities)*time_interval), reversed/float(len(velocities)*time_interval), len(moving)/float(len(velocities)), numpy.mean(numpy.abs(moving)),numpy.mean(moving)]

# print(calculate_speed(centroids, 1, calibration))
# print(calculate_velocities(centroids, 1, calibration, normal))
# print(measure_movement_properties(velocities, threshold, time_interval))




build_data_dictionary()
build_centroids()

fig, axs = plt.subplots(2)
fig.tight_layout(pad=3.0)


build_velocity_plot()
build_speed_plot()


plt.show();


