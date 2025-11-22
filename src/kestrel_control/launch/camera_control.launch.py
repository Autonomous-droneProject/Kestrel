import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # get the path to the config files
    pid_gains_path = os.path.join(
        get_package_share_directory('kestrel_control'), 'config', 'pid_gains.yaml'
    )
    servo_params_path = os.path.join(
        get_package_share_directory('kestrel_control'), 'config', 'servo_params.yaml'
    )

    # node that handles dynamic camera tracking
    dynamic_camera_control_node = Node(
        package='kestrel_control',
        executable='dynamic_camera_control_node',
        name='dynamic_camera_control_node',
        parameters=[pid_gains_path, servo_params_path],
        remappings=[
            ('odom', '/mavros/local_position/pose'),  # connecting to real drone position!!!
        ]
    )

    servo_bridge_node = Node(
        package='kestrel_control',
        executable='servo_to_mavros_bridge',
        name='servo_to_mavros_bridge'
    )

    return LaunchDescription([
        dynamic_camera_control_node,
        servo_bridge_node
    ])
