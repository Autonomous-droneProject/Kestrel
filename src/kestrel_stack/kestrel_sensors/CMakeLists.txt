cmake_minimum_required(VERSION 3.8)
project(kestrel_sensors)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tca9548a REQUIRED)
find_package(vl53l1x REQUIRED)
# <<< ADDED: Required for generating services/messages
find_package(rosidl_default_generators REQUIRED)

# <<< ADDED: This block generates C++ headers from your .srv file
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SelectChannel.srv"
)

# --- TCA Manager Node ---
add_executable(tca_manager_node src/tca_manager_node.cpp)
# <<< MODIFIED: Correct dependencies for this node
ament_target_dependencies(tca_manager_node
  "rclcpp"
  "tca9548a"
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
# <<< ADDED: Link the executable to the found tca9548a library
target_link_libraries(tca_manager_node tca9548a::tca9548a)


# --- VL53L1X Sensor Node ---
# <<< ADDED: Define the second executable for the sensor
add_executable(vl53l1x_sensor_node src/vl53l1x_sensor_node.cpp)
# <<< ADDED: Define dependencies for the sensor node
ament_target_dependencies(vl53l1x_sensor_node
  "rclcpp"
  "sensor_msgs"
  "vl53l1x"
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
# <<< ADDED: Link the executable to the found vl53l1x library
target_link_libraries(vl53l1x_sensor_node vl53l1x::vl53l1x)


# --- Installation ---
# <<< MODIFIED: Added the second executable to the install rule
install(TARGETS
  tca_manager_node
  vl53l1x_sensor_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()