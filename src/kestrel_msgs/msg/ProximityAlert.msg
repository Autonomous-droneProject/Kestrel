# Warns when something gets too close

# Header for timestamp and frame ID

std_msgs Header header

# Distance to the closest detected object

float32 Distance

# Severity level of the alert

uint8 SEVERITY_LOW 1 # CAUTION
uint8 SEVERITY_MEDIUM # WARNING
uint8 SEVERITY_HIGH # CRITICAL

uint8 severity