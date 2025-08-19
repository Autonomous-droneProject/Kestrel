import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # get the path to the gazebo_ros package
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # get the path to your custom packages
    pkg_kestrel_sim = get_package_share_directory('kestrel_simulation')
    pkg_kestrel_perception = get_package_share_directory('kestrel_perception')
    pkg_kestrel_control = get_package_share_directory('kestrel_control')
    pkg_kestrel_communication = get_package_share_directory('kestrel_communication')

    # start gazebo with an empty world
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    # spawn the maze world model into gazebo
    spawn_maze = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'maze_model',
            '-file', os.path.join(pkg_kestrel_sim, 'models', 'maze_model', 'model.sdf'),
            '-x', '0',
            '-y', '0',
            '-z', '0'
        ],
        output='screen'
    )

    # spawn the kestrel drone model into gazebo
    spawn_drone = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'kestrel_drone_lidar',
            '-file', os.path.join(pkg_kestrel_sim, 'models', 'kestrel_drone_lidar', 'model.sdf'),
            '-x', '0',
            '-y', '-8', # start inside the maze entrance
            '-z', '1.0'
        ],
        output='screen'
    )

    # launch the entire perception stack
    launch_perception_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_kestrel_perception, 'launch', 'perception_full.launch.py')
        )
    )

    # launch the entire control stack
    launch_control_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_kestrel_control, 'launch', 'control_stack.launch.py')
        )
    )

    # launch the entire communication stack
    launch_communication_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_kestrel_communication, 'launch', 'communication.launch.py')
        )
    )

    return LaunchDescription([
        start_gazebo,
        spawn_maze,
        spawn_drone,
        launch_perception_stack,
        launch_control_stack,
        launch_communication_stack
    ])
