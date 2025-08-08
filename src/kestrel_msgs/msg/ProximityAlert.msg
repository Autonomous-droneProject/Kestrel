# Warns when something gets too close

# Header for timestamp and frame ID

std_msgs/Header header

# Distance to the closest detected object

float32 distance

# Severity level of the alert

uint8 SEVERITY_LOW  = 1 # CAUTION
uint8 SEVERITY_MEDIUM = 2 # WARNING
uint8 SEVERITY_HIGH = 3 # CRITICAL

uint8 severity