import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # ok mavros has its own extensive launch files
    # it's best practice to include them rather than trying to recreate the node from scratch
    # im gonna use apm.launch.py since we ARE connecting to an ardupilot controller after all
    mavros_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mavros'), 'launch', 'apm.launch')
        ),
        launch_arguments={
            'fcu_url': os.path.join(
                get_package_share_directory('kestrel_control'), 'config', 'ardupilot_interface.yaml'
            )
        }.items()
    )

    return LaunchDescription([
        mavros_launch
    ])