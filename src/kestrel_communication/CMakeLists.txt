#treating kestrel_communication as a CMake-based C++ package that also installs Python scripts.

cmake_minimum_required(VERSION 3.10)
project(kestrel_communication)

# Core build dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(kestrel_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(rclpy REQUIRED)

# Include directory for headers
include_directories(include)

# Build the C++ node(s)
add_executable(base_station_node src/base_station_node.cpp)
add_executable(emergency_handler src/emergency_handler.cpp)
add_executable(failsafe_manager src/failsafe_manager.cpp)
add_executable(heartbeat_monitor src/heartbeat_monitor.cpp)
add_executable(telemetry_manager src/telemetry_manager.cpp)
ament_target_dependencies(base_station_node "rclcpp" "std_msgs" "kestrel_msgs")
ament_target_dependencies(emergency_handler "rclcpp" "std_msgs" "kestrel_msgs")
ament_target_dependencies(failsafe_manager "rclcpp" "std_msgs" "sensor_msgs" "kestrel_msgs")
ament_target_dependencies(heartbeat_monitor "rclcpp" "std_msgs" "kestrel_msgs")
ament_target_dependencies(telemetry_manager "rclcpp" "sensor_msgs" "kestrel_msgs" "mavros_msgs")

# Install C++ executables
install(TARGETS
  base_station_node
  emergency_handler
  failsafe_manager
  heartbeat_monitor
  telemetry_manager
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# Install Python module (in python/kestrel_communication)
install(DIRECTORY python/kestrel_communication
  DESTINATION lib/python3.12/site-packages/
)

# Install standalone scripts
install(PROGRAMS
  scripts/command_sender.py
  scripts/emergency_test.py
  scripts/telemetry_viewer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
