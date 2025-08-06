#!/bin/bash

# ============ Kestrel ROS 2 Python Node Environment Setup ============ #
# This script sets up the virtual environment and ROS 2 overlay
# so the Python node vision_node.py runs with correct dependencies.

# --- 1. Activate the virtual environment --- #
if [ -d "kestrel_env" ]; then
  echo "[kestrel] Activating virtual environment: kestrel_env"
  source kestrel_env/bin/activate
else
  echo "[kestrel] Virtual environment 'kestrel_env' not found."
  echo "         Run: python3 -m venv kestrel_env && source kestrel_env/bin/activate"
  exit 1
fi

# --- 2. Source ROS 2 Jazzy --- #
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  echo "[kestrel] Sourcing ROS 2 Jazzy"
  source /opt/ros/jazzy/setup.bash
else
  echo "[kestrel] ROS 2 Jazzy not found at /opt/ros/jazzy"
  echo "         Make sure ROS 2 is installed"
  exit 1
fi

# --- 3. Source the workspace install/overlay --- #
if [ -f "install/setup.bash" ]; then
  echo "[kestrel] Sourcing colcon workspace: install/setup.bash"
  source install/setup.bash
else
  echo "[kestrel] Workspace not built yet. Run: colcon build"
  exit 1
fi

# --- 4. Fix PYTHONPATH for virtualenv + ROS compatibility --- #
export PYTHONPATH="$PWD/kestrel_env/lib/python3.12/site-packages:$PYTHONPATH"
echo "[kestrel] PYTHONPATH updated for virtualenv"

# --- 5. Done --- #
echo "[kestrel] Environment ready. You can now run: ros2 run kestrel_vision vision_node"
