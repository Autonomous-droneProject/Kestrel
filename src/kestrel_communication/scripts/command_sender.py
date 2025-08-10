import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # get the path to this package's config directory
    config_dir = os.path.join(get_package_share_directory('kestrel_communication'), 'config')

    # node that handles the main interface with the ground station
    base_station_node = Node(
        package='kestrel_communication',
        executable='base_station_node',
        name='base_station_node',
        parameters=[os.path.join(config_dir, 'comms_params.yaml')]
    )

    # node that monitors for critical events and triggers immediate emergency stops
    emergency_handler_node = Node(
        package='kestrel_communication',
        executable='emergency_handler',
        name='emergency_handler',
        parameters=[os.path.join(config_dir, 'emergency_triggers.yaml')]
    )

    # node that aggregates various sensor data into standard telemetry messages
    telemetry_manager_node = Node(
        package='kestrel_communication',
        executable='telemetry_manager',
        name='telemetry_manager',
        parameters=[os.path.join(config_dir, 'telemetry_config.yaml')]
    )

    # node that monitors the connection to the ground station
    heartbeat_monitor_node = Node(
        package='kestrel_communication',
        executable='heartbeat_monitor',
        name='heartbeat_monitor',
        parameters=[os.path.join(config_dir, 'comms_params.yaml')]
    )

    # node that handles soft failsafe actions like rtl or land
    failsafe_manager_node = Node(
        package='kestrel_communication',
        executable='failsafe_manager',
        name='failsafe_manager',
        parameters=[os.path.join(config_dir, 'failsafe_actions.yaml')]
    )

    return LaunchDescription([
        base_station_node,
        emergency_handler_node,
        telemetry_manager_node,
        heartbeat_monitor_node,
        failsafe_manager_node
    ])