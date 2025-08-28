# Use base image we just compiled
FROM l4t-ros2:jazzy

USER root

# Set a working directory inside the container
WORKDIR /app

# Copy scripts into directory
COPY setup ./setup/
RUN chmod -R +x ./setup/

USER $USER

# Install Kestrel
RUN ./setup/src_install/clone_ws.sh
# Run install ros dependencies
RUN ./setup/src_install/install_build_depend.sh
