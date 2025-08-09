# the output data and health status from a VL53L1X TOF sensor array

# header for timestamp and frame id
std_msgs/Header header

# array of distance measurements from the sensors roi 
# values are in millimeters
uint16[] distance_mm

# status of each corresponding sensor reading in the distance_mm array
# the arrays MUST be the same size
uint8[] sensor_status

# status constants
uint8 SENSOR_STATUS_OK = 0
uint8 SENSOR_STATUS_ERROR = 1
uint8 SENSOR_STATUS_NOT_READY = 2

