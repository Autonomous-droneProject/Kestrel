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

    # Trajectory controller for smooth path following (Pure Pursuit 3D)
    # This runs in parallel with WaypointManager - choose which to use via topic remapping
    # To use: remap 'trajectory/setpoint' to 'mavros/setpoint_raw/local'
    traj_controller_node = Node(
        package='kestrel_control',
        executable='traj_controller_node',
        name='traj_controller',
        output='screen',
        parameters=[
            os.path.join(kestrel_control_share, 'config', 'traj_controller.yaml')
        ],
        remappings=[
            ('planning/path', 'planning/path'),
            ('odometry/local_pose', 'odometry/local_pose'),
            # Default: publish to separate topic (parallel with WaypointManager)
            # Change to 'mavros/setpoint_raw/local' to use as primary controller
            ('trajectory/setpoint', 'trajectory/setpoint'),
        ]
    )

    return LaunchDescription([
        ardupilot_translator_node,
        command_validator_node,
        frame_transformer_node,
        # Uncomment the line below to enable trajectory controller
        # traj_controller_node,
    ])