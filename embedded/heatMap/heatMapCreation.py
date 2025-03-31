#Visualizing and interpreting sensor data

import matplotlib.pyplot as plt
import numpy as np
import subprocess as subP

#Start the C++ compiled subprocess
process = subP.Popen(['C:/Users/Tyler Jackson/01 - SCHOOL/01-KESTREL/Model/embedded/heatMap/sensorReading.exe'], stdout=subP.PIPE, text=True)

#Create a numpy array that will be populated with the values from the heatmap
heatmap_data = np.zeros((10,10))

#Enable interactive plot mode
plt.ion()
#Create your plot
fix,ax = plt.subplots()

while True:
    #Read a line from the compiled C++ code
    line = process.stdout.readline()
    
    if(line):
        
        #Convert the csv string to list of floats:
        # Create a list:
        # Map each string value to float:
        # Strip white space then split at each comma:
        sensor_values = list(map(float, line.strip().split(',')))

        #Update the first row of heatmap data
        heatmap_data[0, :len(sensor_values)] = sensor_values

        #Update visualization
        ax.imshow(heatmap_data, cmap='hot', interpolation='nearest')
        plt.pause(0.001)
    else:
        break

