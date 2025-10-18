import unittest
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import String, Empty, Header
from kestrel_msgs.msg import ObstacleGrid
import subprocess
import sys
import threading
from builtin_interfaces.msg import Time
import numpy as np

from rclpy.time import Time
from rclpy.clock import Clock
import math


def set_header(msg, node, frame_id="map"):
    msg.header.frame_id = frame_id
    msg.header.stamp = node.get_clock().now().to_msg()
    return msg

def make_sphere(a, b, c, r, resolution=20):
    """
    Generate (x, y, z) points on a sphere centered at (a, b, c) with radius r.

    Args:
        a, b, c (float): Center of the sphere
        r (float): Radius of the sphere
        resolution (int): Number of steps for theta and phi (controls density)

    Returns:
        list[tuple[float, float, float]]: Points on the sphere surface
    """
    points = []
    for i in range(resolution + 1):
        theta = math.pi * i / resolution  # polar angle (0 to pi)
        for j in range(resolution * 2 + 1):
            phi = 2 * math.pi * j / (resolution * 2)  # azimuthal angle (0 to 2pi)
            x = int(a + r * math.sin(theta) * math.cos(phi))
            y = int(b + r * math.sin(theta) * math.sin(phi))
            z = int(c + r * math.cos(theta))
            if x >= 0 and y >= 0 and z >= 0:
                points.append((x, y, z))
    return points


def make_wall_zy(x, y, z, size, resolution=20):
    """
    Generate (x, y, z) points on a flat wall lying on the ZY-plane (x is constant).

    Args:
        x, y, z (float): Center of the wall
        size (float): Half-size (total height/width = 2*size)
        resolution (int): Number of points along each axis

    Returns:
        list[tuple[int, int, int]]: Points on the wall surface
    """
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            py = int(y - size + 2 * size * (i / resolution))  # vertical (Y)
            pz = int(z - size + 2 * size * (j / resolution))  # depth (Z)
            px = int(x)  # constant X value
            if px >= 0 and py >= 0 and pz >= 0:
                points.append((px, py, pz))
    return points

def make_line(x, y, z, size):
    points = []
    for i in range(size + 1):
        for j in range(size + 1):
            points.append((x, y+i, j))
    return points 

class PlannerTest(Node):
    def __init__(self):
        super().__init__('planner_test_node')

        self.received_path = None
        self.current_planner_status = None
        self.status_history = []
        self.path_count = 0
        self.last_path_time = None
        self.current_waypoint = None
        self.waypoint_count = 0

        self.obstacle_grid_pub = self.create_publisher(ObstacleGrid, 'perception/obstacle_grid', 10)
        self.local_pose_pub = self.create_publisher(PoseStamped, 'odometry/local_pose', 10)
        self.setpoint_pub = self.create_publisher(PositionTarget, 'mavros/setpoint_raw/local', 10)
        self.replan_pub = self.create_publisher(Empty, 'planning/replan', 10)
        self.path_sub = self.create_subscription(Path, 'planning/path', self.path_callback, 10)
        self.planner_status_sub = self.create_subscription(String, 'planning/planner_status', self.planner_status_callback, 10)
        self.waypoint_sub = self.create_subscription(PoseStamped, 'planning/waypoint', self.waypoint_callback, 10)

    def path_callback(self, msg):
        """Callback for path messages from pathing node"""
        self.path_count += 1
        self.last_path_time = time.time()
        self.get_logger().info(f"Received path #{self.path_count} with {len(msg.poses)} waypoints")
        self.received_path = msg

    def planner_status_callback(self, msg):
        """Callback for planner status messages"""
        self.current_planner_status = msg.data
        self.status_history.append((msg.data, time.time()))
        print(f"[Planner Status] {msg.data}")

    def waypoint_callback(self, msg):
        """Callback for waypoint messages from waypoint manager"""
        self.waypoint_count += 1
        self.current_waypoint = msg
        self.get_logger().info(f"Received waypoint #{self.waypoint_count}: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")

    def reset_tracking(self):
        """Reset all tracking variables for new test scenarios"""
        self.received_path = None
        self.path_count = 0
        self.last_path_time = None
        self.status_history = []
        self.current_waypoint = None
        self.waypoint_count = 0
    

class TestKestrelPlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pathing_process = subprocess.Popen(
            ["ros2", "run", "kestrel_planning", "kestrel_planning"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        cls.waypoint_process = subprocess.Popen(
            ["ros2", "run", "kestrel_planning", "kestrel_waypoint"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def print_stream(stream, prefix):
            for line in iter(stream.readline, ''):
                sys.stdout.write(f"[{prefix}] {line}")
            stream.close()

        cls.pathing_stdout_thread = threading.Thread(
            target=print_stream, args=(cls.pathing_process.stdout, "PATHING-OUT"), daemon=True
        )
        cls.pathing_stderr_thread = threading.Thread(
            target=print_stream, args=(cls.pathing_process.stderr, "PATHING-ERR"), daemon=True
        )
        cls.waypoint_stdout_thread = threading.Thread(
            target=print_stream, args=(cls.waypoint_process.stdout, "WAYPOINT-OUT"), daemon=True
        )
        cls.waypoint_stderr_thread = threading.Thread(
            target=print_stream, args=(cls.waypoint_process.stderr, "WAYPOINT-ERR"), daemon=True
        )

        cls.pathing_stdout_thread.start()
        cls.pathing_stderr_thread.start()
        cls.waypoint_stdout_thread.start()
        cls.waypoint_stderr_thread.start()

        time.sleep(3)

        rclpy.init()
        cls.node = PlannerTest()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up processes and ROS"""
        cls.pathing_process.terminate()
        cls.waypoint_process.terminate()
        cls.pathing_process.wait(timeout=5)
        cls.waypoint_process.wait(timeout=5)
        rclpy.shutdown()
    
    def create_obstacle_grid(self, width=100, height=100, depth=50, obstacles=None):
        grid_msg = ObstacleGrid()
        grid_msg.header = Header()
        grid_msg = set_header(grid_msg, self.node, "map")
        
        grid_msg.width = width
        grid_msg.height = height
        grid_msg.depth = depth
        grid_msg.resolution = 1.0
        
        grid_msg.origin.position.x = 0.0
        grid_msg.origin.position.y = 0.0
        grid_msg.origin.position.z = 0.0
        grid_msg.origin.orientation.w = 1.0
        
        total_cells = width * height * depth
        grid_msg.data = [0] * total_cells 
        
        if obstacles:
            for obstacle in obstacles:
                ox, oy, oz = obstacle  
                idx = oz * (width * height) + oy * width + ox
                grid_msg.data[idx] = 100  
        
        return grid_msg

    def create_position_target(self, x, y, z, frame_id="map"):
        """Create PositionTarget message for goal setting"""
        target = PositionTarget()
        target.header = Header()
        target = set_header(target, self.node, frame_id)
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        target.type_mask = PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ | \
                          PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | \
                          PositionTarget.IGNORE_YAW_RATE
        target.position.x = x
        target.position.y = y
        target.position.z = z
        target.yaw = 0.0
        return target
    
    def create_pose_stamped(self, x, y, z, frame_id="map"):
        """Create PoseStamped message for current pose"""
        pose = PoseStamped()
        pose.header = Header()
        pose = set_header(pose, self.node, frame_id)
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        return pose

    def wait_for_subscribers(self, timeout=100):
        start_time = time.time()
        while (self.node.obstacle_grid_pub.get_subscription_count() == 0 or
               self.node.local_pose_pub.get_subscription_count() == 0 or
               self.node.setpoint_pub.get_subscription_count() == 0) and time.time() - start_time < timeout:
            print(f"waiting for subscribers {time.time() - start_time}")
            rclpy.spin_once(self.node, timeout_sec=0.1)
        print("topics subscribed")
        

    def wait_for_planner_status(self, expected_status, timeout=10):
        start_time = time.time()
        while self.node.current_planner_status != expected_status and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.node.current_planner_status == expected_status

    def wait_for_path(self, timeout=10):
        initial_count = self.node.path_count
        start_time = time.time()
        while self.node.path_count <= initial_count and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
           
        return self.node.path_count > initial_count

    def wait_for_waypoint(self, timeout=100):
        initial_count = self.node.waypoint_count
        start_time = time.time()
        while self.node.waypoint_count <= initial_count and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        end_time = time.time()
        print(f"took {end_time - start_time} connect")
        return self.node.waypoint_count > initial_count

    def test_pathing_node_basic_planning(self):
        print("Test #1")
        
        self.wait_for_subscribers()
        self.node.reset_tracking()
        obstacles = make_line(0, 15, 0, 5)
        #obstacles = [(10,10, 0), (10,13, 5), (5,5, 0), (24,24,4), (10,10,1), (19,19,2)]
        obstacle_grid = self.create_obstacle_grid(width=50, height=50, depth=30, obstacles=obstacles)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        print("sent obstacle grid")
        time.sleep(0.5)

        current_pose = self.create_pose_stamped(0.0, 0.0, 0.0)
        self.node.local_pose_pub.publish(current_pose)
        print("sent current pose")
        time.sleep(0.5)

        goal = self.create_position_target(0.0, 45.0, 0.0)
        self.node.setpoint_pub.publish(goal)
        print("sent goal")
        time.sleep(1.0)

        replan_msg = Empty()
        self.node.replan_pub.publish(replan_msg)
        print("triggered replan")
        time.sleep(0.5)
        
        path_received = self.wait_for_path(timeout=500)
        
        self.assertTrue(path_received, "No path received from planner")
        self.assertIsNotNone(self.node.received_path, "Path is None")
        self.assertGreater(len(self.node.received_path.poses), 0, "Path is empty")

        current_pose = self.create_pose_stamped(0.0, 45.0, 0.0)
        self.node.local_pose_pub.publish(current_pose)


        print(f"Basic planning successful: {len(self.node.received_path.poses)} waypoints")
        print("testing a replan")
        goal = self.create_position_target(0.0, 0.0, 0.0)
        self.node.setpoint_pub.publish(goal)
        print("sent goal")
        time.sleep(1.0)


        path_received = self.wait_for_path(timeout=500)
        self.assertTrue(path_received, "No path received from planner")
        self.assertIsNotNone(self.node.received_path, "Path is None")
        self.assertGreater(len(self.node.received_path.poses), 0, "Path is empty")



    


if __name__ == '__main__':
    unittest.main(verbosity=2)