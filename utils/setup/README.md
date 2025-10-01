# Setup Instructions
This specific set of scripts was meant to work on wsl. This may not work for other operating systems.

To run these bash scripts, first, download the repository to your local machine (non-wsl). Then, from wsl, navigate to where this folder is on your local machine and run the scripts from there.

## Initial installation
These instructions assume you are navigated to the non-wsl repo through /mnt/.
1. If user has not been configured and you are in root, run './ros_install/new_user.sh'
2. Set permissions up using 'sudo ./ros_install/setup_permissions.sh
If you have ran any of these instructions, you will need to restart WSL.
3. From here, run './main.sh'
4. Enjoy!

## Script order list
1. ros_install/new_user.sh
2. ros_install/setup_perimssions.sh
3. ros_install/ros2_base.sh
4. src_install/clone_ws.sh
5. src_install/install_build_depend.sh
6. src_build/build_add_swap.sh
7. src_build/build_colcon_workspace.sh
8. sim_install/gazebo_install.sh
9. sim_install/setup_ardupilot_sitl.sh

