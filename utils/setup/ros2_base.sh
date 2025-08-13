#!/bin/bash

# This script automates the installation of ROS 2 Jazzy on a fresh
# Ubuntu 24.04 LTS (Noble Numbat) installation inside WSL.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting ROS 2 Jazzy installation script..."

# 1. Update and upgrade system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Set up locale
# ROS 2 requires a UTF-8 locale.
echo "Setting up locale..."
sudo apt install locales -y
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
echo "Locale setup complete. Verification:"
locale

# 3. Add the ROS 2 apt repository
echo "Adding ROS 2 APT repository..."
sudo apt install software-properties-common -y
sudo add-apt-repository universe -y

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 4. Install ROS 2 packages and development tools
echo "Updating apt list and installing ROS 2 packages..."
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools

# 5. Environment setup
echo "Sourcing ROS 2 setup file and adding to .bashrc..."
source /opt/ros/jazzy/setup.bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

echo "ROS 2 Jazzy installation and setup complete!"

# 6. Verify the installation
echo "Verifying installation with a demo talker and listener..."
echo "Open a new terminal and run 'ros2 run demo_nodes_cpp talker' and 'ros2 run demo_nodes_py listener' in another."
echo "You should see messages being published and received."