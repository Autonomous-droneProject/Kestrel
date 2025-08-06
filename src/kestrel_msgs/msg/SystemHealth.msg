# Report on how well all systems are functioning

# Header for timestamp
std_msgs/Header header

# Overall system status lole

uint8 STATUS_OK = 0
uint STATUS_WARN 1
uint STATUS_ERROR = 2
uint STATUS_CRITICAL = 3
uint8 status

# CPU and Memory usage
float32 cpu_usage_percent
float32 memory_usage_percent

# System temperature IN CELSIUS!!
float32 temperature_celsius

# Ok so this is a list of key-value pairs for other component-specific status lol
diagnostic_msgs/KeyValue[]