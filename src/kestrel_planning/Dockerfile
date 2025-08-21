FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV ROS_DOMAIN_ID=0

# Install tools needed for building & rosdep
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    git \
    curl \
    lsb-release \
    build-essential \
    python3-rosdep \
 && rm -rf /var/lib/apt/lists/*

# Update rosdep (init usually already done in ROS docker images)
RUN rosdep update || true

# Copy your workspace
COPY . /ros_ws
WORKDIR /ros_ws

# Install dependencies declared in package.xml
RUN rosdep install --from-paths src --ignore-src -r -y


