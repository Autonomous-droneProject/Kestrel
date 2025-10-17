RUNNING SITL CONTAINER WITH QGC (old)

Prerequisites:
    - Docker
    - Qgroundcontrol

sitl Dockerfile creates an image with a multistage build that allows you to run Ardupilot's sitl.

It's meant to be run with QGC. The container has an entrypoint that starts sim_vehicle.py for
ArduCopter.

To build (assuming you're in the same directory as the Dockerfile):

    docker  build -t  ardupilot-sitl .

To get it working with QGC, install QGC on your machine (if on Windows, install inside WSL). Then, get your machine's local IP address (on linux):
    
    ip addr show eth0 | grep inet

Then pass that IP address as an argument to the Docker container:

    docker run -it --rm  ardupilot-sitl --out=<your-IP-address>:14551

Note:
    -it starts an interactive shell, so you can interact with the SITL terminal
    --rm removes the container after it stops running.

You could technically choose a port other than 14551, this is just what I've decided to use.

On QGC, configure the link to receive UDP packets:

Vehicle Configurations (top left corner) -> communication links (click on configure) 
-> Add new link -> type=UDP -> port=14551 (or whichever one you chose as output) 
-> server addresses=<your-IP-address>

Click on save and then connect.

Once you have both QGC setup and the docker container running, you should be able to
connect to SITL with the base station.

NEW: RUNNING SITL WITH GAZEBO

gazebo: 
    docker build -t gazebo-harmonic -f Dockerfile.gazebo .
    docker run -it --rm --name gz --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gazebo-harmonic

SITL:
    docker build -t sitl -f Dockerfile.sitl .
    docker run -it --network host --rm --name sitl  sitl --out=172.18.215.158:14551

Issues: 
the bridge works when they are on the host network, but not when they are on their own private network.

Possible reason:
SITL is trying to send messages with address 127.0.0.1, but that address refers to the container itself, so
it doesn't reach to the gazebo container. When they are both on the same host network, they share the namespace so
it in fact works. However, we want them to run on their own network to isolate the stack from the host's network. It's not
reliable to use host network as its functionality may differ in diferent platforms.

Note: ardupilot_gazebo and SITL have a bidirectional communication, the plugin uses the following arguments.
- fcu_addr: This parameter specifies the address (IP or hostname) of the Flight Control Unit (FCU). It's used by the plugin to send actuator commands back to the SITL.

- fdm_addr: This parameter specifies the address (IP or hostname) of the Flight Dynamics Model (FDM). It's used by the plugin to receive sensor data from the SITL.

TODO:
    - test running containers in macOS, determine whether to use a separate network
    - if a new network is required, research how to get the private network going.

SOLUTION: make the containers share the same namespace.
    To Run:
        gazebo: docker run -it --rm --name gz -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gazebo-harmonic
        sitl: docker run -it --network=container:gz --rm --name sitl  sitl --out=172.18.215.158:14551


Issue: QGC cant be connected. Network issues between the containers and the host.

Solution: 
    - in Docker Desktop enable host network
    - SITL now connects to the host
    - Run QGC on the host

If you're on Windows, install QGC on windows, not WSL.
Make a link in QGC for udp connection on port 14551

Docker compose yaml file mounts build and install directories in docker so that builds are persistent

The first time you run the containers you must colcon build from the root of the workspace (kestre/)
there is a chance you run into dependency issues due to rosdep not installing everything in the build face,
if that is the case, do the following on the root of the workspace:
    - rosdep install --from-paths src --ignore-src -r -y

then rebuild:
    - colcon build

The containers will start up a bash shell, to run:
    docker compose up -d
On separate terminals:
    docker compose attach gazebo
    docker compose attach sitl

How to run a world with the containers
On the gazebo terminal:
    gz sim -v4 -r <name-of-sdf-file>

This starts gazebo with the world file provided

On the sitl terminal:
    sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out=udp:127.0.0.1:14551 --map --console

This starts sitl with an output port 14551 for QGC

Because of the way the code is setup, you can modify the source code in your IDE and the changes will automatically
be passed to the container, you can then build and run the code inside the container so you can essentially treat it as a full
development environment. 

you can open multiple terminals by doing 
docker compose exec gazebo

you can do this to launch multiple packages together.