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

# Testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  
  # Create a library from waypoint_manager_node without the main function
  # We'll compile the source with a flag to exclude main
  add_library(waypoint_manager_lib
    src/waypoint_manager_node.cpp
  )
  
  target_compile_definitions(waypoint_manager_lib PRIVATE EXCLUDE_MAIN)
  
  target_include_directories(waypoint_manager_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  
  ament_target_dependencies(waypoint_manager_lib
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
  
  # Create a library from pathing_node without the main function
  add_library(pathing_node_lib
    src/pathing_node.cpp
    src/dstar_planner.cpp
  )
  
  target_compile_definitions(pathing_node_lib PRIVATE EXCLUDE_MAIN)
  
  target_include_directories(pathing_node_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  
  ament_target_dependencies(pathing_node_lib
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
  
  # Build test executable for waypoint manager
  ament_add_gtest(test_waypoint_manager_node
    test/test_waypoints.cpp
  )
  
  target_include_directories(test_waypoint_manager_node PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  
  # Link test with the library (not the source file directly)
  target_link_libraries(test_waypoint_manager_node
    waypoint_manager_lib
    pthread
  )
  
  ament_target_dependencies(test_waypoint_manager_node
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
  
  # Build test executable for pathing node
  ament_add_gtest(test_pathing_node
    test/test_pathing_node.cpp
  )
  
  target_include_directories(test_pathing_node PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  
  target_link_libraries(test_pathing_node
    pathing_node_lib
    pthread
  )
  
  ament_target_dependencies(test_pathing_node
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
  
  install(TARGETS
    test_waypoint_manager_node
    test_pathing_node
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()