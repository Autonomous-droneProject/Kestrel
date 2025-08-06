cmake_minimum_required(VERSION 3.10)
project(kestrel_planning)

# Required ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Include directory
include_directories(include)

# Source files
set(SOURCE_FILES
  src/pathing_node.cpp
  src/dstar_planner.cpp
  src/collision_checker.cpp
  src/waypoint_manager.cpp
)

add_executable(pathing_node ${SOURCE_FILES})
ament_target_dependencies(pathing_node rclcpp std_msgs geometry_msgs)

# Install the executable
install(TARGETS pathing_node
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

# Add GTest unit tests
ament_add_gtest(test_dstar test/test_dstar.cpp)
ament_add_gtest(test_collision_checker test/test_collision_checker.cpp)

ament_package()
