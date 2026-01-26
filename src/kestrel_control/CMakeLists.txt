cmake_minimum_required(VERSION 3.10)
project(kestrel_control)

# Core build dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(kestrel_msgs REQUIRED)
find_package(control_toolbox REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(tf2 REQUIRED)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/usr/share/cmake/geographiclib")
find_package(GeographicLib REQUIRED)

# Include directory for headers
include_directories(include)

# Create library for reusable classes
add_library(kestrel_control_lib
  src/command_validator.cpp
  src/frame_transformer.cpp
  src/traj_controller.cpp
)

# Link lib to dependencies
ament_target_dependencies(kestrel_control_lib
  rclcpp
  geometry_msgs
  nav_msgs
  kestrel_msgs
  sensor_msgs
  mavros_msgs
  control_toolbox
  tf2
  GeographicLib
)

# C++ node executables
add_executable(ardupilot_translator_node src/ardupilot_translator_node.cpp)
add_executable(command_validator_node src/command_validator_node.cpp)
add_executable(dynamic_camera_control_node src/dynamic_camera_control_node.cpp)
add_executable(frame_transformer_node src/frame_transformer_node.cpp)
add_executable(servo_to_mavros_bridge src/servo_to_mavros_bridge.cpp)
add_executable(traj_controller_node src/traj_controller_node.cpp)

# Link dependencies
#ament_target_dependencies(ardupilot_translator_node "rclcpp" "geometry_msgs" "mavros_msgs")
#ament_target_dependencies(command_validator "rclcpp" "geometry_msgs")
#ament_target_dependencies(dynamic_camera_control_node "rclcpp" "geometry_msgs" "kestrel_msgs" "control_toolbox")
#ament_target_dependencies(frame_transformer rclcpp "rclcpp" "sensor_msgs" "geometry_msgs" "GeographicLib")
#target_link_libraries(frame_transformer GeographicLib::GeographicLib)
target_link_libraries(ardupilot_translator_node PUBLIC kestrel_control_lib)
target_link_libraries(command_validator_node PUBLIC kestrel_control_lib)
target_link_libraries(dynamic_camera_control_node PUBLIC kestrel_control_lib)
target_link_libraries(frame_transformer_node PUBLIC kestrel_control_lib)
target_link_libraries(servo_to_mavros_bridge PUBLIC kestrel_control_lib)
target_link_libraries(traj_controller_node PUBLIC kestrel_control_lib)



# Install all compiled binaries
install(TARGETS
  ardupilot_translator_node
  command_validator_node
  dynamic_camera_control_node
  frame_transformer_node
  servo_to_mavros_bridge
  traj_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers (if any)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# Install Python module (if you end up using it)
install(DIRECTORY python/kestrel_control
  DESTINATION lib/python3.12/site-packages/
)

ament_add_gtest(test_command_validator test/test_command_validator.cpp)
ament_add_gtest(test_frame_transformer test/test_frame_transformer.cpp)
target_link_libraries(test_command_validator kestrel_control_lib)
target_link_libraries(test_frame_transformer kestrel_control_lib)

ament_package()
