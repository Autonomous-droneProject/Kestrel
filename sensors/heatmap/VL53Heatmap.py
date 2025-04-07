import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import serial
import time

# Initialize the serial port (make sure COM5 is available and no other app is using it)
sensorSerialData = serial.Serial(port='COM5', baudrate=9600, timeout=1)

size = 4

# Create grid for the heatmap (using size+1 points for pcolormesh)
x, y = np.meshgrid(np.linspace(0, size, size+1), np.linspace(0, size, size+1))
distance_matrix = np.zeros((size, size))

fig, ax2 = plt.subplots(figsize=(6,6))
quad1 = ax2.pcolormesh(x, y, distance_matrix, vmin=0, vmax=3000, shading='flat')
bar = fig.colorbar(quad1, ax=ax2)
bar.set_label("Distance (mm)")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

def init():
    #quad1.set_array(np.array([]))
    return quad1,

def animate(i):
    global distance_matrix  # ensure we refer to the global variable
    if sensorSerialData.in_waiting > 0:
        try:
            # Read one line from serial
            data_line = sensorSerialData.readline().decode('utf-8').strip()
            # Example: "536,764,1009,...,75,"
            if data_line:
                # Split on commas and remove any empty strings
                parts = [p for p in data_line.split(',') if p]
                if len(parts) == size*size:
                    distances = [float(val) for val in parts]
                    # Reshape into 4x4 matrix
                    distance_matrix = np.reshape(distances, (size, size))
                    
                    # Apply any reordering if necessary
                    # For example: if you want to flip the matrix vertically:
                    distance_matrix = np.flipud(distance_matrix)
                    
                    # Optionally, apply interpolation or smoothing here if desired
                else:
                    print("Incomplete data received:", data_line)
        except Exception as e:
            print("Error processing data:", e)

    # Update heatmap: pcolormesh expects the flattened array for the mesh
    quad1.set_array(distance_matrix.ravel())
    return quad1,

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, blit=False, repeat=False)
plt.show()
