import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

# ok so this file only starts the nodes related to safety and emergency monitoring

def generate_launch_description():

    config_dir = os.path.join(get_package_share_directory('kestrel_communication'), 'config')

    emergency_handler_node = Node(
        package='kestrel_communication',
        executable='emergency_handler',
        name='emergency_handler',
        parameters=[os.path.join(config_dir, 'emergency_triggers.yaml')]
    )

    heartbeat_monitor_node = Node(
        package='kestrel_communication',
        executable='heartbeat_monitor',
        name='heartbeat_monitor',
        parameters=[os.path.join(config_dir, 'comms_params.yaml')]
    )

    failsafe_manager_node = Node(
        package='kestrel_communication',
        executable='failsafe_manager',
        name='failsafe_manager',
        parameters=[os.path.join(config_dir, 'failsafe_actions.yaml')]
    )

    return LaunchDescription([
        emergency_handler_node,
        heartbeat_monitor_node,
        failsafe_manager_node
    ])
