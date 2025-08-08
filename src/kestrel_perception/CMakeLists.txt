cmake_minimum_required(VERSION 3.8)
project(kestrel_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(kestrel_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Include headers
include_directories(include)

# All your perception sources
set(SOURCE_FILES
  src/sensor_fusion_node.cpp
  src/i2c_driver.cpp
  src/uart_driver.cpp
  src/sensor_validator.cpp
  src/spatial_grid.cpp
)

add_executable(sensor_fusion_node ${SOURCE_FILES})
ament_target_dependencies(sensor_fusion_node rclcpp std_msgs sensor_msgs kestrel_msgs rclcpp_components tf2_ros tf2_geometry_msgs)

# Install the node binary
install(TARGETS sensor_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch and config files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/
)

# Install tests
ament_add_gtest(test_sensor_fusion test/test_sensor_fusion.cpp)
ament_target_dependencies(test_sensor_fusion rclcpp ament_cmake_gtest kestrel_msgs)

ament_package()
