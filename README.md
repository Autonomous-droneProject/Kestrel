**Project Kestrel** is an ambitious, student-led initiative focused on building an **autonomous drone system** capable of real-time tracking, pathfinding, and obstacle avoidance!
By integrating advanced computer vision, machine learning, and robotics frameworks, the project aims to push the boundaries of what autonomous aerial systems can achieve in dynamic environments.
**We plan to follow a group of Person Electric Vehicle Riders**, going at an approximate speed of **30mph**, while maintaining autonomous control for a **15-minute flight duration**.

### Install
```
git clone --recurse-submodules https://github.com/Autonomous-droneProject/Kestrel.git
```

### Key Features:
- **Target Tracking:**
   - Uses object detection (YOLO) and multi-object tracking (DeepSORT) to follow a designated subject in real time.
- **Autonomous Navigation:**
   - Employs a custom waypoint pathing algorithm integrated with ArduPilot to adjust flight paths while avoiding obstacles dynamically.
- **Sensor Fusion:**
   - Combines camera input with IR & TOF sensor modules to create a diverse tracking and avoidance system
- **Simulation:**
   - SITL and Gazebo are used for a closed loop simulation for virtual validation before live drone deployment.
- **System Overview:**
   - The Drone will be using ROS 2 Jazzy to create communicable sub systems.


**Project Kestrel** is led by a team of over 30 students across multiple disciplines, Kestrel exemplifies collaborative innovation, many multidisciplinary majors such as:
- Computer Engineering & Science
- Mechanical Engineering
- Electrical Engineering
- And MORE!

Furthermore, **Project Kestrel** is proudly backed by major supporters, including **Blue Origin**, and is actively supported by several leading student organizations at UCF such as:
- Association of Computing Machinery (**ACM**)
- Knight Hacks (**KH**)
- **IEEE**
- **AI@UCF**

The project timeline targets the **start of prototyping in early to mid-July**, with a transition to **real drone integration by the beginning of August**.
~~-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------~~
