cmake_minimum_required(VERSION 3.10)
project(kestrel_planning)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(kestrel_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)

include_directories(include)

# Create executables with all source files included directly
add_executable(kestrel_planning
  src/pathing_node.cpp
  src/dstar_planner.cpp
)

add_executable(kestrel_waypoint
  src/waypoint_manager_node.cpp
)

# Set dependencies for executables
ament_target_dependencies(kestrel_planning
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  kestrel_msgs
  mavros_msgs
)

ament_target_dependencies(kestrel_waypoint
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  kestrel_msgs
  mavros_msgs
)

# Install executables
install(TARGETS 
  kestrel_planning
  kestrel_waypoint
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()