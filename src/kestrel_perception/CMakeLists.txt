cmake_minimum_required(VERSION 3.10)
project(kestrel_perception)

# Core ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Include headers
include_directories(include)

# All your perception sources
set(SOURCE_FILES
  src/sensor_fusion_node.cpp
  src/i2c_driver.cpp
  src/uart_driver.cpp
  src/clustering_node.cpp
  src/sensor_validator.cpp
  src/spatial_grid.cpp
)

add_executable(sensor_fusion_node ${SOURCE_FILES})
ament_target_dependencies(sensor_fusion_node rclcpp std_msgs sensor_msgs)

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
ament_add_gtest(test_clustering test/test_clustering.cpp)
ament_add_gtest(test_i2c test/test_i2c_driver.cpp)

ament_package()
