import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    kestrel_control_share = get_package_share_directory('kestrel_control')
    
    # arduPilot translator node (with new emergency services!!)
    ardupilot_translator_node = Node(
        package='kestrel_control',
        executable='ardupilot_translator_node',
        name='ardupilot_translator_node',
        output='screen'
    )
    
    # command validator for safety checks
    command_validator_node = Node(
        package='kestrel_control',
        executable='command_validator_node',
        name='command_validator_node',
        output='screen',
        parameters=[{
            'safety.max_linear_speed': 2.0,
            'safety.max_angular_speed': 1.5,
            'safety.geofence.min_x': -100.0,
            'safety.geofence.max_x': 100.0,
            'safety.geofence.min_y': -100.0,
            'safety.geofence.max_y': 100.0,
            'safety.geofence.min_z': 0.5,
            'safety.geofence.max_z': 10.0
        }]
    )
    
    # frame transformer for GPS conversion
    frame_transformer_node = Node(
        package='kestrel_control',
        executable='frame_transformer_node',
        name='frame_transformer_node',
        output='screen'
    )
    
    return LaunchDescription([
        ardupilot_translator_node,
        command_validator_node,
        frame_transformer_node
    ])