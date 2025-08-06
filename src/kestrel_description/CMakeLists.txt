# Updated from ROS 1 format to ROS 2.

cmake_minimum_required(VERSION 3.10)
project(kestrel_description)

find_package(ament_cmake REQUIRED)

# Install resource folders like urdf/, meshes/, config/, launch/
foreach(dir config launch meshes urdf)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    install(DIRECTORY ${dir}/
      DESTINATION share/${PROJECT_NAME}/${dir})
  endif()
endforeach()

ament_package()
