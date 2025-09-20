#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from builtin_interfaces.msg import Time


def set_header(msg, frame_id="map"):
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(sec=int(time.time()))
    return msg


class DStarPublisher(Node):
    def __init__(self):
        super().__init__('dstar_publisher')

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.start_pub = self.create_publisher(PoseStamped, '/current_pose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.replan_pub = self.create_publisher(Empty, '/replan', 10)

        self.get_logger().info("DStarPublisher ready to publish test data...")

    def create_map(self, width=50, height=50, obstacle_coords=None):
        msg = OccupancyGrid()
        msg.info.width = width
        msg.info.height = height
        msg.info.resolution = 1.0
        msg.data = [0] * (width * height)  # free space

        if obstacle_coords:
            for (ox, oy) in obstacle_coords:
                if 0 <= ox < width and 0 <= oy < height:
                    idx = oy * width + ox
                    msg.data[idx] = 100  # obstacle

        return set_header(msg, "map")

    def create_pose(self, x, y, z=0.0):
        msg = PoseStamped()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        return set_header(msg, "map")

    def publish_test_sequence(self):
        self.get_logger().info("Publishing initial empty map...")
        map_msg = self.create_map()
        self.map_pub.publish(map_msg)
        time.sleep(1.0)

        self.get_logger().info("Publishing start and goal poses...")
        self.start_pub.publish(self.create_pose(0.0, 0.0, 0.0))
        self.goal_pub.publish(self.create_pose(30.0, 30.0, 0.0))
        time.sleep(1.0)

        self.get_logger().info("Publishing map with obstacle at (10,10)...")
        map_msg = self.create_map(obstacle_coords=[(10, 10)])
        self.map_pub.publish(map_msg)
        time.sleep(1.0)

        self.get_logger().info("Publishing replan command...")
        self.replan_pub.publish(Empty())
        time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = DStarPublisher()

    try:
        while rclpy.ok():
            node.publish_test_sequence()
            time.sleep(5.0)  # wait before repeating sequence
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
