Autonomous Drone Tracking System
Main Repository for Multi-Object Tracking, Embedded Control, and Autonomous Navigation

📦 Project Structure
This repository serves as the central compilation point for an autonomous drone system capable of tracking targets using computer vision, managing flight paths, and avoiding obstacles in real time.

🗂 Folder Breakdown
1. deepSORT/
Contains:

main.py for initializing and running the deepSORT pipeline

data_association.py: Implements a CNN-based feature extractor and association logic

2. SORT/
Contains:

main.py for initializing and running the SORT pipeline

data_association.py: Simpler distance-based association mechanism

3. sharedModules/
Modules shared between both SORT and deepSORT:

kalman_filter.py: Predicts and updates object motion

track_management.py: Handles object lifecycle (creation, confirmation, deletion)

yolo_input.py: Interfaces with YOLOv7 for real-time object detection and bounding box generation

4. Docker/
Docker configuration for consistent deployment across environments

✈ Embedded & Flight Control Subsystem
In addition to the vision modules, this system includes a hardware integration layer to allow full autonomous drone functionality.

Components:
flight_control/ (or in a future embedded/ directory)
Autopilot Engine: Computes dynamic waypoints and issues movement commands based on object tracking feedback.

Path Planning: Incorporates algorithms (A*, Dijkstra, RRT) to generate viable flight paths toward targets while avoiding known obstacles.

Sensor Fusion Module: Fuses data from LiDAR, ultrasonic, and camera inputs to create a 3D environmental map.

Obstacle Avoidance: Uses real-time telemetry to dynamically avoid unexpected obstacles.

Motor Interface: Sends direct motor or ESC control signals via a microcontroller (e.g., Raspberry Pi, STM32).

🔄 System Flow
YOLOv7 detects objects of interest from the drone’s camera feed.

SORT / deepSORT processes detections:

Kalman Filter predicts object motion

Data Association matches current detections with previous tracks

Track Manager updates track status

Shared Modules (YOLO, KF, Track Management) allow clean modularity between models.

Autopilot Engine receives positional data and calculates navigational adjustments.

Embedded Controllers interpret movement directives and adjust motors accordingly.

Sensor Fusion & Obstacle Avoidance ensure safe, smooth operation even in complex environments.

✅ Features
🔍 Real-Time Multi-Object Tracking (SORT / deepSORT)

🧠 Modular CNN Integration for Improved Association (deepSORT)

🛩 Autonomous Flight with Live Positional Feedback

🛠 Embedded Integration (Motor Control, Path Adjustment)

🧭 Adaptive Obstacle Avoidance using LiDAR

🐳 Dockerized for Portability & Reproducibility

📌 Notes
Designed to track dynamic targets (e.g., cyclists, runners) in open or semi-cluttered environments.

deepSORT is ideal for more robust environments due to its improved pattern recognition via CNN.

Both models are fully compatible with the shared YOLO input module and Kalman-based tracking system.

Full integration with drone sensors and microcontroller enables in-field deployment.


