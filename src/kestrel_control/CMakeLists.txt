cmake_minimum_required(VERSION 3.10)
project(kestrel_control)

# Core build dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Include directory for headers
include_directories(include)

# C++ node executables
add_executable(ardupilot_translator_node src/ardupilot_translator_node.cpp)
add_executable(command_validator src/command_validator.cpp)
add_executable(dynamic_camera_control_node src/dynamic_camera_control_node.cpp)
add_executable(frame_transformer src/frame_transformer.cpp)

# Link dependencies
ament_target_dependencies(ardupilot_translator_node rclcpp std_msgs)
ament_target_dependencies(command_validator rclcpp std_msgs)
ament_target_dependencies(dynamic_camera_control_node rclcpp std_msgs)
ament_target_dependencies(frame_transformer rclcpp std_msgs)

# Install all compiled binaries
install(TARGETS
  ardupilot_translator_node
  command_validator
  dynamic_camera_control_node
  frame_transformer
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers (if any)
install(DIRECTORY include/
  DESTINATION include/
)

# Install Python module (if you end up using it)
install(DIRECTORY python/kestrel_control
  DESTINATION lib/python3.12/site-packages/
)

ament_package()
