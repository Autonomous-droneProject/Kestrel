#!/bin/bash

# This script automates the dependency installation process for the
# ROS 2 workspace located in ~/Kestrel.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Set the ROS 2 distribution
ROS_DISTRO="jazzy"
# Set the workspace directory
WORKSPACE_DIR="$HOME/Kestrel"
# Set the virtual environment directory
VENV_DIR="$HOME/kestrel_venv"
# --- End of Configuration ---

echo "--------------------------------------------------------"
echo "  ROS 2 Dependency Installer for Kestrel"
echo "--------------------------------------------------------"

# 1. Navigate to the workspace directory
echo "Navigating to the workspace directory: $WORKSPACE_DIR"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Error: Directory '$WORKSPACE_DIR' not found."
    echo "Please ensure you have cloned the repository correctly."
    exit 1
fi
cd "$WORKSPACE_DIR"

# 2. Source the main ROS 2 environment
echo "Sourcing ROS 2 $ROS_DISTRO environment..."
source "/opt/ros/$ROS_DISTRO/setup.bash"

# Check if the sourcing was successful.
if [ $? -ne 0 ]; then
    echo "Error: Failed to source ROS 2 $ROS_DISTRO setup file. Please check the path."
    exit 1
fi

# 3. Setup Python virtual environment and install system packages
echo "Setting up Python virtual environment and installing system packages..."
sudo apt update
sudo apt install -y \
  python3-pip \
  python3.12-venv \
  ros-jazzy-geographic-msgs \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# 4. Install Python dependencies using pip
echo "Installing Python dependencies with pip..."
pip install \
  jinja2 \
  pyyaml \
  typeguard \
  'setuptools<80' \
  catkin_pkg \
  empy \
  numpy \
  Cython \
  colcon-common-extensions

# 5. Initialize and update rosdep
echo "Initializing and updating rosdep database..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# 6. Install ROS 2 dependencies
echo "Installing all ROS 2 dependencies from packages in 'src'..."
# The rosdep command will still work from the workspace directory
sudo rosdep install --rosdistro $ROS_DISTRO --from-paths src --ignore-src -r -y
deactivate

echo "--------------------------------------------------------"
echo "  Dependency installation and virtual environment setup complete."
echo "--------------------------------------------------------"
echo "Note: The virtual environment now exists at '$VENV_DIR'."
echo "To activate it in new terminals, run: source $VENV_DIR/bin/activate"
echo "To deactivate it, run: deactivate"
echo "--------------------------------------------------------"