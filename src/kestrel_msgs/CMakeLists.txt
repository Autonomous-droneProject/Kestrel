message(STATUS "PACKAGE_XML_PATH=${CMAKE_CURRENT_SOURCE_DIR}/package.xml")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.xml" PKGXML)
message(STATUS "XML_CONTENT=${PKGXML}")

cmake_minimum_required(VERSION 3.10)
project(kestrel_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
#find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraCommand.msg"
  "msg/FlightStatus.msg"
  "msg/Track.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/TrackCenter.msg"
  "msg/ObstacleGrid.msg"
  "msg/ProximityAlert.msg"
  "msg/SensorArray.msg"
  "msg/SensorReading.msg"
  "msg/SystemHealth.msg"
  "msg/TrackArray.msg"
  "srv/CalibrateCamera.srv"
  "srv/GetSystemStatus.srv"
  "srv/ResetSensors.srv"
  "srv/SetFlightMode.srv"
  "srv/TriggerEmergencyStop.srv"
  "action/FollowTarget.action"
  "action/NavigateToWaypoint.action"
  DEPENDENCIES std_msgs geometry_msgs diagnostic_msgs#vision_msgs
)

ament_export_dependencies(rosidl_default_runtime diagnostic_msgs)
ament_package()