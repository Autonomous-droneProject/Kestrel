#!/bin/bash

# Get a list of all ROS 2 packages
packages=$(ros2 pkg list)

# Loop through each package and list its executables
for package in $packages
do
    echo "Executables in package: $package"
    ros2 pkg executables "$package"
    echo "" # Add a blank line for readability
done