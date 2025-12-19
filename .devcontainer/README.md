This devcontainer builds the `l4t-ros2:jazzy` image using the provided `Dockerfile` and starts a container with the same runtime options used by the project's documentation.

Before opening the folder in the devcontainer, allow X connections from the host by running on your host machine:

```
xhost +
```

Then open the repository in VS Code and choose "Reopen in Container". The container is built from `.devcontainer/Dockerfile` and is started with GPU support, host networking, DISPLAY forwarded, and the host's `$HOME` bind-mounted so GUI and ROS tooling work similarly to the manual `docker run` command.

The devcontainer will now run `scripts/rosdep_install.sh` once after the container is created to install ROS package dependencies from the workspace `src/` directory using `rosdep`.

To re-run the installer manually inside the container:

```bash
# from inside the container or via 'devcontainer exec'
bash $HOME/Kestrel/scripts/rosdep_install.sh
```
