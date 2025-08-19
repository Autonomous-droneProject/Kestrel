# Represents a 3D map showing where obstacles are in space

# Header for timestamp and frame ID
std_msgs/Header header

# Grid metadata

float32 resolution # Resolution of the grid in meters/cell
uint32 width # self explanitory
uint32 height # self explanitory
uint32 depth #depth of the grid

# Origin of the grid [m] in the frame given in the Header
geometry_msgs/Pose origin

# The grid data is stored in a 1D array in a row-major order lol
# 0 means free, 100 means occupied, -1 means unknown
int8[] data