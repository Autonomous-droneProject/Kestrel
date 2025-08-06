cmake_minimum_required(VERSION 3.8)
project(kestrel_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tca9548a REQUIRED)
find_package(vl53l1x REQUIRED)

message(STATUS "[DEBUG] tca9548a include directories are: ${tca9548a_INCLUDE_DIRS}")
message(STATUS "[DEBUG] vl53l1x include directories are: ${vl53l1x_INCLUDE_DIRS}")

include_directories(include)

add_executable(sensors_drivers_node src/sensors_drivers_node.cpp)

ament_target_dependencies(sensors_drivers_node
  rclcpp
  sensor_msgs
  tca9548a
  vl53l1x
)

install(TARGETS
  sensors_drivers_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()