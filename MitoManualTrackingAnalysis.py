import numpy as np
from matplotlib.pylab import plt #load plot library
import csv #imports csv library


data_dict = {} #dictionary--> pairwise, key is mitchondrial track (1,2,3 ect.) Value is a two-pule list of x values and list of y values
list_of_rows = [] #list of rows


with open('MitoGfp200323.csv', newline='') as csvfile:
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


def plot_data():
	fig, axs = plt.subplots(2)
	fig.tight_layout(pad=3.0)

	for key in sorted(data_dict.keys()):
		X_data = data_dict[key][0]
		Y_data = data_dict[key][1]
		axs[0].plot(X_data,Y_data)

	axs[0].set_xlabel('X-coordinate')
	axs[0].set_xlim(0,250)
	axs[0].set_ylabel('Y-coordinate')
	axs[0].set_ylim(250,0)
	axs[0].set_title('Mitochondrial Tracking')


	for key in sorted(data_dict.keys()):
		X_data = data_dict[key][0]
		Y_data = data_dict[key][1]
		if (determine_direction(Y_data[0],Y_data[-1])):
			axs[1].plot(X_data,Y_data, 'b')
			print('Blue ', key)
		else:
			axs[1].plot(X_data,Y_data, 'y')
			print('Gold', key)

	axs[1].set_xlabel('X-coordinate')
	axs[1].set_xlim(0,250)
	axs[1].set_ylabel('Y-coordinate')
	axs[1].set_ylim(250,0) #Inverts the y-axis to match the numbering system in Fiji Manual Tracking
	axs[1].set_title('Mitochondrial Directionality')



	plt.show()
	


build_data_dictionary() #Calls build_data_dictionary function
plot_data() #Calls plot_data function
