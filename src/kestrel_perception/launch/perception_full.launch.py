import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():

    # declare a launch argument so i can easily enable/disable the gui!!!
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # get the path to the URDF file
    urdf_path = os.path.join(
        get_package_share_directory('kestrel_description'),
        'urdf',
        'kestrel_description.urdf'
    )

    # Read the URDF file content directly
    with open(urdf_path, 'r') as infp:
        robot_description_content = infp.read()

    # ok so this node is going to read the URDF
    # andd also publish the sensor transform to /tf_static
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time
        }]
    )

    # main sensor fusion node n stuff
    sensor_fusion_node = Node(
        package='kestrel_perception',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        output='screen',
        # ok aldem you can load the parameters from the yaml file here
        # ONCE YOU ADD IT!!
        parameters=[
            {'world_frame_id': 'base_link'},
            {'proximity_alert_distance': 1.5}
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        robot_state_publisher_node,
        sensor_fusion_node
    ])