# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kestrel is an autonomous drone system built on ROS 2 Jazzy for tracking groups of PEV (Personal Electric Vehicle) riders at ~30mph. The system uses YOLO for object detection, DeepSORT for tracking, and D* Lite for pathfinding.

## Build Commands

```bash
# Build all packages (run from workspace root containing src/)
colcon build

# Build specific package
colcon build --packages-select <package_name>

# Clean build
rm -rf build/ install/ log/ && colcon build
```

## Testing

```bash
# Run all tests
colcon test

# Run tests for specific package
colcon test --packages-select <package_name>

# Run with console output
colcon test --event-handlers console_direct+

# View test results
colcon test-result --verbose
```

**Test frameworks:**
- C++ packages: Google Test via `ament_cmake_gtest`
- Python packages: pytest via `ament_cmake_pytest`

## Development Environment

Use the devcontainer (`.devcontainer/`) which provides the Jetson L4T + ROS 2 Jazzy environment. After opening in the container, dependencies are installed automatically via `scripts/rosdep_install.sh`.

For manual dependency installation:
```bash
bash scripts/rosdep_install.sh
```

## Architecture

### ROS 2 Package Structure

**C++ Packages (C++17):**
- `kestrel_planning` - D* Lite pathfinding with octree voxel costmap
- `kestrel_control` - Flight control, command validation, geofence enforcement, ArduPilot translation
- `kestrel_perception` - Sensor fusion (ToF, I2C, UART sensors), obstacle detection
- `kestrel_communication` - Base station interface, telemetry, emergency handling

**Python Packages:**
- `kestrel_vision` - YOLO11 object detection
- `kestrel_tracking` - DeepSORT multi-object tracking
- `kestrel_clustering` - Groups PEV detections into centroids

**Shared:**
- `kestrel_msgs` - Custom ROS 2 messages, services, and actions
- `kestrel_description` - URDF robot model

**Hardware Drivers (git submodules in `src/driver/`):**
- `vl53l1x` - ToF sensor
- `tca9548a` - I2C multiplexer
- `ina228` - Power monitor
- `tfluna` - Distance sensor

### Data Flow

1. **Vision Pipeline:** Camera → `kestrel_vision` (YOLO) → `kestrel_tracking` (DeepSORT) → `kestrel_clustering` → Track centroids
2. **Planning:** Obstacle data + goal → `kestrel_planning` (D* Lite) → Path
3. **Control:** Path + sensor data → `kestrel_control` → ArduPilot commands via MAVRos
4. **Perception:** ToF/distance sensors → `kestrel_perception` → Obstacle grid + proximity alerts

### Key Message Types (kestrel_msgs)

- `ObjectTrack` - Tracked object with ID
- `TrackCentroid` - Group centroid position
- `ObstacleGrid` - 3D obstacle representation
- `ProximityAlert` - Collision warning
- `FlightStatus` - Current flight state
- Services: `TriggerEmergencyStop`, `SetFlightMode`
- Actions: `FollowTarget`, `NavigateToWaypoint`

## Launch Files

Located in each package's `launch/` directory:
- `kestrel_control/launch/control_stack.launch.py` - Full control system
- `kestrel_perception/launch/perception_full.launch.py` - All sensors
- `kestrel_communication/launch/communication.launch.py` - Comms system
- `simulation/launch/sim_*.launch.py` - Gazebo simulation environments

## Code Style

- C++: Compiler flags `-Wall -Wextra -Wpedantic`
- Python: Linted with flake8 and pep257 (via ament linting)

## CI/CD

GitHub Actions workflow (`.github/workflows/test.yaml`) builds a Docker image and runs `colcon build`. Triggered manually via workflow_dispatch.
