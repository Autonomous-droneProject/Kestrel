cmake_minimum_required(VERSION 3.10)
project(kestrel_communication)

# Core build dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(kestrel_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(rclpy REQUIRED)

# Include directory for headers
include_directories(include)

# Build lib
add_library(kestrel_communication_lib
  src/base_station.cpp
  src/emergency_handler.cpp
  src/failsafe_manager.cpp
  src/heartbeat_monitor.cpp
  src/telemetry_manager.cpp
)

ament_target_dependencies(kestrel_communication_lib
  rclcpp
  std_msgs
  sensor_msgs
  mavros_msgs
  kestrel_msgs
)

add_executable(base_station_node src/base_station_node.cpp)
add_executable(emergency_handler_node src/emergency_handler_node.cpp)
add_executable(failsafe_manager_node src/failsafe_manager_node.cpp)
add_executable(heartbeat_monitor_node src/heartbeat_monitor_node.cpp)
add_executable(telemetry_manager_node src/telemetry_manager_node.cpp)
target_link_libraries(base_station_node PUBLIC kestrel_communication_lib)
target_link_libraries(emergency_handler_node PUBLIC kestrel_communication_lib)
target_link_libraries(heartbeat_monitor_node PUBLIC kestrel_communication_lib)
target_link_libraries(failsafe_manager_node PUBLIC kestrel_communication_lib)
target_link_libraries(telemetry_manager_node PUBLIC kestrel_communication_lib)

# Install C++ executables
install(TARGETS
  base_station_node
  emergency_handler_node
  failsafe_manager_node
  heartbeat_monitor_node
  telemetry_manager_node
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


ament_add_gtest(test_base_station test/test_base_station.cpp)
ament_add_gtest(test_emergency_handler test/test_emergency_handler.cpp)
target_link_libraries(test_base_station kestrel_communication_lib)
target_link_libraries(test_emergency_handler kestrel_communication_lib)
#target_link_libraries(test_base_station "rclcpp" "std_msgs" "kestrel_msgs")
#target_link_libraries(test_emergency_handler "rclcpp" "std_msgs" "kestrel_msgs")


ament_package()
