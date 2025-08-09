import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # get the path to this package
    kestrel_control_share = get_package_share_directory('kestrel_control')

    # include the mavros bridge launch file
    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kestrel_control_share, 'launch', 'mavros_bridge.launch.py')
        )
    )

    # include the camera control launch file
    camera_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kestrel_control_share, 'launch', 'camera_control.launch.py')
        )
    )

    # node to convert gps data to a local coordinate frame
    frame_transformer_node = Node(
        package='kestrel_control',
        executable='frame_transformer',
        name='frame_transformer'
    )

    # node to validate flight commands for safety
    command_validator_node = Node(
        package='kestrel_control',
        executable='command_validator',
        name='command_validator',
        parameters=[os.path.join(kestrel_control_share, 'config', 'safety_params.yaml')]
    )

    # node to translate high-level waypoints into low-level mavlink commands
    ardupilot_translator_node = Node(
        package='kestrel_control',
        executable='ardupilot_translator_node',
        name='ardupilot_translator_node'
    )

    return LaunchDescription([
        mavros_launch,
        camera_control_launch,
        frame_transformer_node,
        command_validator_node,
        ardupilot_translator_node
    ])