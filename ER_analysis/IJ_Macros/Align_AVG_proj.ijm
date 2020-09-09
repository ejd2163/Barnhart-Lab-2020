/*
 * Macro template to process multiple images in a folder
 */

//~USER INPUT~
//input_dir must be a directory containing images of either flash responses or grating responses
input_dir = "C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi9_strong/2s_flashes/"; 
//output_dir must be a 'grating' or 'flashes' folder 
output_dir = "C:/Users/vymak_i7/Desktop/New_ER/Screen/Mi9_strong/moving_dark_bar/" 
//press Ctrl+R to run the entire code 

//i=12 for 2s_flashes and MLB
//counter = 0 for 10s_flashes

//set global variable (counter) to 0 outside of macro; this is target_list's counter for processFolder macro
var counter=10
//run Macro
processFolder(input_dir, output_dir);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input, output) {
	//list files in your input and output directories
	source_list = getFileList(input);
	source_list = Array.sort(source_list);
	target_list = getFileList(output);
	target_list = Array.sort(target_list);
	//run the AAA code for ea/ .tif file in your input directory 
	for (i = 10; i < 11; i++) { //source_list.length
		if (endsWith(source_list[i], "/")){
	    	print((counter+1) + ") " + source_list[i]); 
	    	print("Processing: " + input+source_list[i]);
	    	print("Saving to: " + output+target_list[counter]);
	    	Align_AVG_proj(input+source_list[i], output+target_list[counter]); 
	    	counter++; //counter++ -> counter = counter + 1
	    }
	    else {
	    	print("Current file "+input+source_list[i]+" is not a folder.");
	    	debug("break");
	    }
	}
}

function Align_AVG_proj(source_dir, target_dir) { //source_dir = flashes; target_dir = gratings 
//(Section 1) OPEN SELECTED IMAGE & AVERAGE PIXEL INTENSITY OF CHANNEL W/ STRONGEST SIGNAL
	//open raw image data
	open(source_dir+"average_projection.tif");
	open(target_dir+"average_projection.tif");
	Channel_A = "flashes";
	Channel_B = "gratings";
	rename(Channel_B+"_avg_proj");
	run("Put Behind [tab]");
	rename(Channel_A+"_avg_proj");
	
//(Section 2) ALIGN FIRST CHANNEL W/ STRONGEST SIGNAL (for TurboReg instructions, visit this website: http://bigwww.epfl.ch/thevenaz/turboreg/)
	setBatchMode(true); //true -> hides newly opened images; streamlines TurboReg more quickly
	//create a registered Channel_A window
	selectWindow(Channel_B+"_avg_proj");
	width = getWidth();
	height = getHeight();
	run("TurboReg ","-align " 
		 + "-window "+Channel_A+"_avg_proj 0 0 "+(width-1)+" "+(height-1)+" "
		 + "-window "+Channel_B+"_avg_proj 0 0 "+(width-1)+ " "+(height-1)+" "
		 + "-rigidBody "
		 + (width/2)+" "+(height/2)+" "+(width/2)+" "+(height/2)+" "//1st landmarks of source & target (gives overall translation)
		 + (width/2)+" "+round(0.15625*height)+" "+(width/2)+" "+round(0.15625*height)+" "//2nd landmarks of source & target (determines rotation angle)
		 + (width/2)+" "+round(0.84375*height)+" "+(width/2)+" "+round(0.84375*height)+" "//3rd landmarks of source & target (determines rotation angle)
		 + "-showOutput"); //"showOutput" in TurboReg triggers "Refined Landmarks" and "Log" windows
	selectWindow("Output"); //output = 2 sequential images for each frame in 'Channel_A': (1) raw data; (2) mask/black background
	close();
	setBatchMode(false);
	//calculate alignment values of Channel A [first frame]
	sourceX0 = getResult("sourceX", 0); // First line of the "Refined Landmarks" table.
	sourceY0 = getResult("sourceY", 0);
	targetX0 = getResult("targetX", 0);
	targetY0 = getResult("targetY", 0);
	sourceX1 = getResult("sourceX", 1); // Second line of the "Refined Landmarks" table.
	sourceY1 = getResult("sourceY", 1);
	targetX1 = getResult("targetX", 1);
	targetY1 = getResult("targetY", 1);
	sourceX2 = getResult("sourceX", 2); // Third line of the "Refined Landmarks" table.
	sourceY2 = getResult("sourceY", 2);
	targetX2 = getResult("targetX", 2);
	targetY2 = getResult("targetY", 2);
	tdx = targetX0 - sourceX0; // translated offset of x-coordinate
	tdy = targetY0 - sourceY0; // translated offset of y-coordinate
	//calculate rotation offset
	dx = sourceX2 - sourceX1;
	dy = sourceY2 - sourceY1;
	sourceAngle = atan2(dy, dx);
	dx = targetX2 - targetX1;
	dy = targetY2 - targetY1;
	targetAngle = atan2(dy, dx);
	rotation = (targetAngle - sourceAngle)*(180.0/ PI); //rotation offset in degrees
	//create an array containing x offset, y offset, and rotation values of the first frame of Channel A
	source_array = newArray(tdx,tdy,rotation); 
	
	//create masks for gratings from masks for flashes using alignment calculations
/*	open(source_dir+"background.tif");
	rename("background");
	open(source_dir+"mask.tif");
	rename("mask");
	open(source_dir+"mask-M1.tif");
	rename("mask-M1");
	open(source_dir+"mask-M5.tif");
	rename("mask-M5");
	open(source_dir+"mask-M9-M10.tif");
	rename("mask-M9-M10");
	masks = newArray("background","mask","mask-M1","mask-M5","mask-M9-M10");
	
/*	open(source_dir+"background.tif");
	rename("background");
  	open(source_dir+"mask.tif");
	rename("mask");
	open(source_dir+"mask-Lo2.tif");
	rename("mask-Lo2");
	open(source_dir+"mask-Lo4.tif");
	rename("mask-Lo4");
	masks = newArray("mask","mask-Lo2","mask-Lo4");
*/	
	open(source_dir+"background.tif");
	rename("background");
	open(source_dir+"mask.tif");
	rename("mask");
	open(source_dir+"mask-M9-M10.tif");
	rename("mask-M9-M10");
	masks=newArray("background","mask","mask-M9-M10");

	for (i = 0; i<=lengthOf(masks)-1; i++) {
		selectWindow(masks[i]);
		x_offset = source_array[0];
		y_offset = source_array[1];
		rotation_angle = source_array[2];
		run("Translate...", "x="+x_offset+" y="+y_offset+" interpolation=None");
		run("Rotate... ", "angle="+rotation_angle+" grid=1 interpolation=None");
		saveAs("Tiff",target_dir+File.separator+masks[i]+".tif");
	}
	run("Close All");
}