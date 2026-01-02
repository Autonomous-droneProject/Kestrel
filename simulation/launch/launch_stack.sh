#!/usr/bin/env bash
echo "Starting Embedded Stack"

# Prevent sourcing
[[ "${BASH_SOURCE[0]}" != "$0" ]] && {
    echo "[ERROR] Do not source this script. Run it as ./script.sh"
    return 1
}

set -m

SCRIPT_PGID=$(ps -o pgid= $$ | tr -d ' ')
LOG_DIR="ros_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" || {
    echo "[ERROR] Failed to create log directory"
    exit 1
}

cleanup() {
    echo -e "\n[INFO] Shutting down... cleaning up $LOG_DIR"
    kill -INT -- -"$SCRIPT_PGID" 2>/dev/null
    sleep 2
    kill -TERM -- -"$SCRIPT_PGID" 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

source /opt/ros/jazzy/setup.bash
source ~/Kestrel/install/setup.bash

echo "[INFO] Logging to directory: $LOG_DIR"

# Launch and prefix output so you know which node is talking
ros2 launch kestrel_control mavros_bridge.launch.py 2>&1 | sed -e "s/^/[mavros_bridge] /" > "$LOG_DIR/mavros_bridge.log" &
ros2 launch kestrel_communication communication.launch.py 2>&1 | sed -e "s/^/[communication] /" > "$LOG_DIR/communication.log" &
ros2 launch kestrel_control control_stack.launch.py 2>&1 | sed -e "s/^/[control] /" > "$LOG_DIR/control_stack.log" &
ros2 launch kestrel_control camera_control.launch.py 2>&1 | sed -e "s/^/[camera] /" > "$LOG_DIR/camera_control.log" &

# Optional: Tail the logs to the terminal so you still see live updates
tail -f "$LOG_DIR"/*.log 2>/dev/null &

echo "[INFO] ROS 2 processes started. Press Ctrl+C to stop."

wait