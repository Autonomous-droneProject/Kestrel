# Data from one sensor (Like distance or temperature)

# Header for timestamp and frame_id of the sensor

std_msgs/Header header

# Type of the sensor (Like ToF and crap)
string sensor_type

# The actual sensor values
float64 value

# Variance of the reading, if available
float64 variance