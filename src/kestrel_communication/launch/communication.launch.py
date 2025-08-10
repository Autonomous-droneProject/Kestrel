import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # get the path to this package
    kestrel_communication_share = get_package_share_directory('kestrel_communication')

    # node that handles the main interface with the ground station
    base_station_node = Node(
        package='kestrel_communication',
        executable='base_station_node',
        name='base_station_node',
        parameters=[os.path.join(kestrel_communication_share, 'config', 'comms_params.yaml')]
    )

    # node that monitors for critical events and triggers emergencies
    emergency_handler_node = Node(
        package='kestrel_communication',
        executable='emergency_handler',
        name='emergency_handler',
        parameters=[os.path.join(kestrel_communication_share, 'config', 'emergency_triggers.yaml')]
    )

    # node that collects various sensor data into standard telemetry messages
    telemetry_manager_node = Node(
        package='kestrel_communication',
        executable='telemetry_manager',
        name='telemetry_manager',
        parameters=[os.path.join(kestrel_communication_share, 'config', 'telemetry_config.yaml')]
    )

    # node that monitors the connection to the ground station
    heartbeat_monitor_node = Node(
        package='kestrel_communication',
        executable='heartbeat_monitor',
        name='heartbeat_monitor',
        parameters=[os.path.join(kestrel_communication_share, 'config', 'comms_params.yaml')]
    )

    return LaunchDescription([
        base_station_node,
        emergency_handler_node,
        telemetry_manager_node,
        heartbeat_monitor_node
    ])