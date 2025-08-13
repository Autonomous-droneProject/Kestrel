#!/bin/bash

# ==============================================================================
# main.sh
# A master script to execute ROS2 and simulation setup scripts.
#
# This script is designed to be run from any directory and will correctly
# locate and execute other scripts within the same 'setup' folder hierarchy.
# All output (stdout and stderr) is sent to the terminal and a log file.
# ==============================================================================

# --- Configuration ---
# Get the absolute path of the directory where this script is located.
# This is crucial for running the script from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Define the log file path, placing it in the same directory as the main script.
LOG_FILE="$SCRIPT_DIR/install_log.log"

# --- Main Script Logic ---
# Redirect all output (stdout and stderr) to the tee command.
# `tee -a` appends to the log file. Remove the `-a` to overwrite.
# `2>&1` redirects standard error to standard output.
exec &> >(tee -a "$LOG_FILE")

# --- Start of Execution ---
echo "=========================================================="
echo "Starting ROS2 and Simulation Setup at $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================================="
echo ""

# --- Run the installation scripts in the specified order ---
# Each script is called using its full path.

echo "--- Running new_user.sh ---"
sudo "$SCRIPT_DIR/ros_install/new_user.sh"
if [ $? -ne 0 ]; then echo "ERROR: new_user.sh failed. Aborting." >&2; exit 1; fi
echo ""

echo "--- Running ros2_base.sh ---"
bash "$SCRIPT_DIR/ros_install/ros2_base.sh"
if [ $? -ne 0 ]; then echo "ERROR: ros2_base.sh failed. Aborting." >&2; exit 1; fi
echo ""

echo "--- Running clone_ws.sh ---"
bash "$SCRIPT_DIR/src_install/clone_ws.sh"
echo ""

echo "--- Running install_build_depend.sh ---"
bash "$SCRIPT_DIR/src_install/install_build_depend.sh"
if [ $? -ne 0 ]; then echo "ERROR: install_build_depend.sh failed. Aborting." >&2; exit 1; fi
echo ""

echo "--- Running build_add_swap.sh ---"
bash "$SCRIPT_DIR/ros_install/build_add_swap.sh"
if [ $? -ne 0 ]; then echo "ERROR: build_add_swap.sh failed. Aborting." >&2; exit 1; fi
echo ""

echo "--- Running build_colcon_workspace.sh ---"
bash "$SCRIPT_DIR/ros_install/build_colcon_workspace.sh"
if [ $? -ne 0 ]; then echo "ERROR: build_colcon_workspace.sh failed. Aborting." >&2; exit 1; fi
echo ""

echo "--- Running gazebo_install.sh ---"
bash "$SCRIPT_DIR/sim_install/gazebo_install.sh"
if [ $? -ne 0 ]; then echo "ERROR: gazebo_install.sh failed. Aborting." >&2; exit 1; fi
echo ""

echo "--- Running setup_ardupilot_sitl.sh ---"
bash "$SCRIPT_DIR/sim_install/setup_ardupilot_sitl.sh"
if [ $? -ne 0 ]; then echo "ERROR: setup_ardupilot_sitl.sh failed. Aborting." >&2; exit 1; fi
echo ""

# --- End of Execution ---
echo "=========================================================="
echo "All installation scripts finished at $(date)"
echo "=========================================================="