import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

# ok so this file only starts the nodes related to safety and emergency monitoring

def generate_launch_description():

    kestrel_communication_share = get_package_share_directory('kestrel_communication')

    emergency_handler_node = Node(
        package='kestrel_communication',
        executable='emergency_handler_node', # fixed: was 'emergency_handler'
        name='emergency_handler_node',
        parameters=[os.path.join(kestrel_communication_share, 'config', 'emergency_triggers.yaml')]
    )

    heartbeat_monitor_node = Node(
        package='kestrel_communication',
        executable='heartbeat_monitor_node', # fixed: was 'heartbeat_monitor'
        name='heartbeat_monitor_node',
        parameters=[os.path.join(kestrel_communication_share, 'config', 'comms_params.yaml')]
    )

    return LaunchDescription([
        emergency_handler_node,
        heartbeat_monitor_node
    ])