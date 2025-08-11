cmake_minimum_required(VERSION 3.10)
project(kestrel_control)

# Core build dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(kestrel_msgs REQUIRED)
find_package(GeographicLib REQUIRED)
find_package(control_toolbox REQUIRED)
find_package(mavros_msgs REQUIRED)

# Include directory for headers
include_directories(include)

# C++ node executables
add_executable(ardupilot_translator_node src/ardupilot_translator_node.cpp)
add_executable(command_validator src/command_validator.cpp)
add_executable(dynamic_camera_control_node src/dynamic_camera_control_node.cpp)
add_executable(frame_transformer src/frame_transformer.cpp)

# Link dependencies
ament_target_dependencies(ardupilot_translator_node "rclcpp" "geometry_msgs" "mavros_msgs")
ament_target_dependencies(command_validator "rclcpp" "geometry_msgs")
ament_target_dependencies(dynamic_camera_control_node "rclcpp" "geometry_msgs" "kestrel_msgs" "control_toolbox")
ament_target_dependencies(frame_transformer rclcpp "rclcpp" "sensor_msgs" "geometry_msgs" "GeographicLib")
target_link_libraries(frame_transformer GeographicLib::GeographicLib)
# Install all compiled binaries
install(TARGETS
  ardupilot_translator_node
  command_validator
  dynamic_camera_control_node
  frame_transformer
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

#ament_add_gtest(test_command_validator test/test_command_validator.cpp)
#ament_add_gtest(test_frame_transformer test/test_frame_transformer.cpp)
#ament_target_dependencies(test_command_validator "rclcpp" "geometry_msgs")
#ament_target_dependencies(test_frame_transformer "rclcpp" "geometry_msgs" "sensor_msgs")


ament_package()
