ros2 run ros_gz_bridge parameter_bridge \
/lidar/down@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/forward@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/forwardl@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45dfl@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45dfr@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45dfrl@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45dfrll@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45drl@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45drr@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45ufl@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45ufr@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45url@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/l45urr@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/left@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/rear@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/right@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
/lidar/up@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan &
ros2 run ros_gz_bridge parameter_bridge \
/lidar/down/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/forward/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/forwardl/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45dfl/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45dfr/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45dfrl/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45dfrll/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45drl/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45drr/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45ufl/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45ufr/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45url/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/l45urr/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/left/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/rear/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/right/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
/lidar/up/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked &
ros2 run ros_gz_image image_bridge /camera/image_raw /camera/image_raw &
ros2 run ros_gz_bridge parameter_bridge /camera/cmd_pitch@std_msgs/msg/Float64@gz.msgs.Double &
ros2 run ros_gz_bridge parameter_bridge /camera/cmd_yaw@std_msgs/msg/Float64@gz.msgs.Double &
