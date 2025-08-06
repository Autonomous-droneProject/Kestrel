# Current state of the drone (flying, landed, etc.)

# Header for timestamp

std_msgs/Header Header

#Flight mode constraints for clarity

uint8 MODE_UNKOWN = 0
uint8 MODE_MANUAL = 1
uint8 MODE_AUTO = 2
uint8 MODE_LANDING = 3
uint8 MODE_EMERGENCY = 4
uint MODE_LANDED = 5

# Current flight mode
uint flight_mode

# Armed state of the vehicle
bool armed 

# Current battery percentage
float32 battery_percent

# Propeller speeds in RPM. The array order SHOULD be consistent.
uint[] propeller_speeds