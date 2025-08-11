import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    base_station_node = Node(
        package='kestrel_communication',
        executable='base_station_node',
        name='base_station_node',
        parameters=[os.path.join(
            get_package_share_directory('kestrel_communication'), 'config', 'comms_params.yaml'
        )]
    )

    return LaunchDescription([
        base_station_node