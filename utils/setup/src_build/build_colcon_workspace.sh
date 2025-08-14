#!/bin/bash

# This script navigates to the ~/Kestrel workspace, builds it, and then
# returns to the original directory. It now includes activation of the
# Python virtual environment.

# Exit immediately if a command exits with a non-zero status.
set -e

# Store the original directory so we can return to it later.
ORIGINAL_DIR=$(pwd)

# Define the workspace and virtual environment directories.
WORKSPACE_DIR="$HOME/Kestrel"
VENV_DIR="$HOME/kestrel_venv"

echo "--------------------------------------------------------"
echo "  ROS 2 Workspace Builder for Kestrel (with venv)"
echo "--------------------------------------------------------"

# 1. Navigate to the workspace directory
echo "Navigating to the workspace directory: $WORKSPACE_DIR"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Error: Directory '$WORKSPACE_DIR' not found."
    echo "Please ensure you have cloned the repository correctly."
    exit 1
fi
cd "$WORKSPACE_DIR"

# 2. Activate the Python virtual environment
echo "Activating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment '$VENV_DIR' not found."
    echo "Please run the dependency installation script first: install_kestrel_depend_venv.sh"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# 3. Source the main ROS 2 environment
# This path is for ROS 2 Jazzy on Ubuntu 24.04. Adjust if needed.
echo "Sourcing ROS 2 Jazzy environment..."
source /opt/ros/jazzy/setup.bash

# Check if the sourcing was successful.
if [ $? -ne 0 ]; then
    echo "Error: Failed to source ROS 2 Jazzy setup file. Please check the path."
    exit 1
fi

# 4. Run colcon build
echo "Running colcon build..."
colcon build \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --continue-on-error

# 5. Source the local workspace setup file
echo "Sourcing the local workspace environment..."
source install/setup.bash

# 6. Navigate back to the original directory
echo "Build complete. Returning to original directory: $ORIGINAL_DIR"
cd "$ORIGINAL_DIR"

echo "--------------------------------------------------------"
echo "  Build successful! You are back in your original location."
echo "--------------------------------------------------------"
echo "To use your packages in a new terminal, remember to:"
echo "  1. Navigate to your workspace: cd ~/Kestrel"
echo "  2. Activate your virtual environment: source $VENV_DIR/bin/activate"
echo "  3. Source the workspace: source install/setup.bash"
echo "--------------------------------------------------------"