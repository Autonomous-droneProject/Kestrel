# Kestrel Simulation Environment

Do you want to test, debug, and develop autonomous drones? Project Kestrel has developed a simulation environment that fully integrates ROS2 Jazzy, Gazebo Harmonic, and ArduPilot's SITL with Docker containers. You can develop code in your favorite IDE, and run it on docker containers.

## Prerequisites

    Qgroundcontrol (QGC)
    Docker desktop
    At least 8GB of RAM
    At least intel i5 or equivalent

Note: if you are in Windows, install QGC for Windows, don't install inside WSL. Also, the hardware requirements are approximations based on the requirements to run Gazebo, they may actually be higher depending on the complexity of the worlds you want to run.

## Before starting

Make sure that host networking is enabled in Docker Desktop. To do that:
    
    Settings->reseources->Network

Check the box for "Enable host networking".

Furthemore, if you are in a macOS machine, you need to look up how to get GUI's with Docker. The following resource might be useful if you're on macOS:
    
    https://youtu.be/cNDR6Z24KLM?si=IVlIC-EfRxIcDXno

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/Autonomous-droneProject/Kestrel.git
   git checkout kestrel-sim-env
   cd Kestrel/simulation/kestrel-sim
   ```

2. Build the Docker images:

   ```bash
   docker compose build
   ```

Note: The first build may take some time as it needs to download and set up all dependencies. Expect it to take around 20 minutes depending on your internet speed.

3. Start the simulation environment:

   ```bash
   docker compose up -d
   ```

This will start both containers: the Gazebo+ROS2 environment and SITL, however, it will only start a bash session for both of them. Instructions for how to run the simulation are below.

4. enter the inside the containers (run in separate terminal windows):

    ```bash
    docker compose attach gazebo-ros2
    docker compose attach sitl
    ```
Because both containers start up a bash session, you can attach to them using the above commands. If you want to enter a new session, you can use:

    docker exec -it  gazebo-ros2 bash
    docker exec -it  sitl bash

## Getting familiar with the gazebo-ros2 container

The gazebo-ros2 container bind mounts the source code of the repository inside the container at `/kestrel/src`. This means that any changes you make to the source code in your host machine will be reflected inside the container. This is useful for development purposes. You can change the source code on your host machine and build it inside the container.

Another thing to note is that the build files are stored inside a Docker volume, so they won't clutter your host machine, and you also don't need to rebuild everything from scratch every time you enter the container. 

The gazebo-ros2 container has all the models, worlds, and launch files included. You can add more models and worlds by adding them to the respective folders in the repository. If this is your first time running the gazebo-ros2 container, you might want to build the workspace first. To do that, run the following commands inside the gazebo-ros2 container:

    cd /kestrel
    colcon build

Note: you always need to source the workspace before running any ROS2 commands. You can do that by running:

    source /kestrel/install/setup.bash

It's possible you might run into errors in the build process due to missing dependencies. If that happens run the following command from the root of the workspace to install the missing dependencies:

    cd /kestrel
    apt update
    rosdep install --from-paths src --ignore-src -r -y

Then try building again.

## Running the simulation

1. Start Gazebo with the desired world:

    ```bash
    gz sim -v4 -r <name-of-sdf-file> 
    ```

the simulation environment includes a variety of worlds located in the `kestrel/simulation/worlds` folder. You can choose any of the sdf files located there. For example, to start the base world, run:

    gz sim -v4 -r baseWorld.sdf

This includes a simple flat world with Kestrel Project's drone model.

2. Start SITL:

    ```bash
    sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14550 --console
    ```
Run this inside the sitl container. It adds an output to UDP port 14551, which is the port that QGroundControl will listen to. Port 14550 is used for mavros connection.

3. Connect QGroundControl (QGC):
    Open QGC on your host machine. It's possible that QGC won't automatically connect to the SITL instance. If that happens, you can manually add a UDP link in QGC by going to:
    
        Settings -> Comm Links -> Add -> UDP -> Set the port to 14551

    Then click "connect", QGC should connect to the SITL instance.

Troubleshooting tip: You need to make sure that docker desktop has host networking enabled. That way your SITL container can share the same namespace as your host machine, allowing QGC to connect to it with address 127.0.0.1

At this point, you should have a running Gazebo simulation with the Kestrel drone model, and QGC connected to the SITL instance. You can now start testing and developing your autonomous drone code! Try telling the drone to take off using QGC, and you should see the drone taking off in the Gazebo simulation.

You can also send commands to the drone from the terminal inside the SITL container using `mavproxy` commands. For example, to arm the drone and take off to an altitude of 10 meters, you can run:

    mode GUIDED
    arm throttle
    takeoff 10

You can also use launch files to start worlds with specific configurations. You can create your own launch files and run them on the gazebo-ros2 container.

## Stopping the simulation

To stop the simulation, you can simply stop the Docker containers by running the following command from the host machine:

    docker compose down

You can also stop individual containers by running:

    docker compose stop gazebo-ros2
    docker compose stop sitl

You can also stop them on Docker Desktop directly.

## Understanding Kestrel ROS2 packages

Below is a list of the main ROS2 packages included in the simulation environment, along with instructions on how to run their nodes.

#### mavros

    ros2 launch mavros apm.launch

Note: Make sure that the GeographicLib datasets are installed. If not, you can install them by running:

    ./kestrel/src/libraries/mavros/mavros/scripts/install_geographiclib_datasets.sh
    cp -r /usr/share/GeographicLib/  /usr/local/share/

#### Perception & Sensor Fusion

    ros2 launch kestrel_perception perception_full.launch.py

#### Communication System
    ros2 launch kestrel_communication communication.launch.py

#### MAVROS Bridge

    ros2 launch kestrel_control mavros_bridge.launch.py

#### Control Nodes

    ros2 run kestrel_control ardupilot_translator_node
    ros2 run kestrel_control command_validator_node
    ros2 run kestrel_control frame_transformer_node

#### Camera Control
    ros2 launch kestrel_control camera_control.launch.py

#### Vision
    ros2 run kestrel_vision vision_node

#### Failsafe Manager
    ros2 run kestrel_communication failsafe_manager_node

#### Path Planning
    ros2 run kestrel_planning kestrel_planning

## Important Notes

- Make sure to always source the workspace before running any ROS2 commands:

      source /kestrel/install/setup.bash

- If you make changes to the source code, remember to rebuild the workspace using

        colcon build

- If you encounter any issues with missing dependencies during the build process, use `rosdep` to install them as described above.

- If you delete the container, you won't lose the volumes where the build files are stored, so you won't need to rebuild everything from scratch.

- If you installed new GeographicLib datasets, make sure to copy them to `/usr/local/share/GeographicLib/` inside the gazebo-ros2 container, as described above. You won't need to do this step again unless you delete the container.
