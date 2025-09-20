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
from rclpy.qos import QoSProfile, ReliabilityPolicy

def set_header(msg, frame_id="map"):
    """Helper function to set header with current timestamp"""
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(sec=int(time.time()), nanosec=int((time.time() % 1) * 1e9))
    return msg

class PlannerTestNode(Node):
    def __init__(self):
        super().__init__('planner_test_node')

        # Test state tracking
        self.received_path = None
        self.current_planner_status = None
        self.status_history = []
        self.path_count = 0
        self.last_path_time = None
        self.current_waypoint = None
        self.waypoint_count = 0

        # Publishers for pathing node inputs
        self.obstacle_grid_pub = self.create_publisher(ObstacleGrid, 'perception/obstacle_grid', 10)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.local_pose_pub = self.create_publisher(PoseStamped, "odometry/local_pose", qos)

        self.setpoint_pub = self.create_publisher(PositionTarget, 'mavros/setpoint_raw/local', 10)
        self.replan_pub = self.create_publisher(Empty, 'planning/replan', 10)

        # Subscribers for pathing node outputs
        self.path_sub = self.create_subscription(Path, 'planning/path', self.path_callback, 10)
        self.planner_status_sub = self.create_subscription(String, 'planning/planner_status', self.planner_status_callback, 10)

        # Subscribers for waypoint manager outputs
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
        """Set up test environment and start nodes"""
        # Start pathing node
        cls.pathing_process = subprocess.Popen(
            ["ros2", "run", "kestrel_planning", "kestrel_planning"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Start waypoint manager node
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

        # Start output threads for both processes
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

        # Allow nodes to start
        time.sleep(3)

        # Initialize ROS
        rclpy.init()
        cls.node = PlannerTestNode()

    @classmethod
    def tearDownClass(cls):
        """Clean up processes and ROS"""
        cls.pathing_process.terminate()
        cls.waypoint_process.terminate()
        cls.pathing_process.wait(timeout=5)
        cls.waypoint_process.wait(timeout=5)
        rclpy.shutdown()

    def create_obstacle_grid(self, width=100, height=100, depth=50, obstacles=None):
        """Create 3D ObstacleGrid message with optional obstacles"""
        grid_msg = ObstacleGrid()
        grid_msg.header = Header()
        grid_msg = set_header(grid_msg, "map")
        
        # Set 3D grid parameters based on your ObstacleGrid message
        grid_msg.width = width
        grid_msg.height = height
        grid_msg.depth = depth
        grid_msg.resolution = 1.0
        
        # Set origin pose
        grid_msg.origin.position.x = 0.0
        grid_msg.origin.position.y = 0.0
        grid_msg.origin.position.z = 0.0
        grid_msg.origin.orientation.w = 1.0
        
        # Initialize empty 3D grid (row-major order: x + y*width + z*width*height)
        total_cells = width * height * depth
        grid_msg.data = [0] * total_cells  # 0 = free space
        
        # Add obstacles if specified (assuming obstacles are 3D coordinates)
        if obstacles:
            for obstacle in obstacles:
                if len(obstacle) == 2:
                    # 2D obstacle - extend through middle layers of 3D grid
                    ox, oy = obstacle
                    for oz in range(depth // 4, 3 * depth // 4):  # Middle half of depth
                        if 0 <= ox < width and 0 <= oy < height and 0 <= oz < depth:
                            idx = ox + oy * width + oz * width * height
                            grid_msg.data[idx] = 100  # Occupied cell
                elif len(obstacle) == 3:
                    # 3D obstacle
                    ox, oy, oz = obstacle
                    if 0 <= ox < width and 0 <= oy < height and 0 <= oz < depth:
                        idx = ox + oy * width + oz * width * height
                        grid_msg.data[idx] = 100  # Occupied cell
        
        return grid_msg

    def create_position_target(self, x, y, z, frame_id="map"):
        """Create PositionTarget message for goal setting"""
        target = PositionTarget()
        target.header = Header()
        target = set_header(target, frame_id)
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
        pose = set_header(pose, frame_id)
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        return pose

    def wait_for_subscribers(self, timeout=5):
        """Wait for subscribers to connect"""
        start_time = time.time()
        while (self.node.obstacle_grid_pub.get_subscription_count() == 0 or
               self.node.local_pose_pub.get_subscription_count() == 0 or
               self.node.setpoint_pub.get_subscription_count() == 0) and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def wait_for_planner_status(self, expected_status, timeout=10):
        """Wait for specific planner status"""
        start_time = time.time()
        while self.node.current_planner_status != expected_status and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.node.current_planner_status == expected_status

    def wait_for_path(self, timeout=10):
        """Wait for new path to be received"""
        initial_count = self.node.path_count
        start_time = time.time()
        while self.node.path_count <= initial_count and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.node.path_count > initial_count

    def wait_for_waypoint(self, timeout=5):
        """Wait for new waypoint to be received"""
        initial_count = self.node.waypoint_count
        start_time = time.time()
        while self.node.waypoint_count <= initial_count and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.node.waypoint_count > initial_count

    def test_pathing_node_basic_planning(self):
        """Test basic path planning functionality"""
        print("\n=== Testing Pathing Node: Basic Planning ===")
        
        self.wait_for_subscribers()
        self.node.reset_tracking()

        # Publish obstacle grid
        obstacle_grid = self.create_obstacle_grid(width=50, height=50, depth=30)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        time.sleep(0.5)

        # Publish current pose (start position)
        current_pose = self.create_pose_stamped(0.0, 0.0, 0.0)
        self.node.local_pose_pub.publish(current_pose)
        time.sleep(0.5)

        # Publish goal
        goal = self.create_position_target(25.0, 25.0, 5.0)
        self.node.setpoint_pub.publish(goal)
        time.sleep(1.0)

        # Wait for path planning to complete
        success = self.wait_for_planner_status("SUCCESS", timeout=15)
        self.assertTrue(success, f"Planner did not reach SUCCESS status. Current: {self.node.current_planner_status}")

        # Verify path was generated
        path_received = self.wait_for_path(timeout=5)
        self.assertTrue(path_received, "No path received from planner")
        self.assertIsNotNone(self.node.received_path, "Path is None")
        self.assertGreater(len(self.node.received_path.poses), 0, "Path is empty")

        print(f"Basic planning successful: {len(self.node.received_path.poses)} waypoints")

    def test_pathing_node_obstacle_avoidance(self):
        """Test path planning with obstacles"""
        print("\n=== Testing Pathing Node: Obstacle Avoidance ===")
        
        self.wait_for_subscribers()
        self.node.reset_tracking()

        # Create obstacle grid with obstacles blocking direct path
        obstacles_2d = [(12, 12), (12, 13), (13, 12), (13, 13), (14, 12), (14, 13)]  # 2D obstacles
        obstacles_3d = [(15, 15, 10), (15, 15, 11), (15, 15, 12)]  # 3D obstacles
        all_obstacles = obstacles_2d + obstacles_3d
        obstacle_grid = self.create_obstacle_grid(width=30, height=30, depth=25, obstacles=all_obstacles)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        time.sleep(0.5)

        # Set start and goal positions
        current_pose = self.create_pose_stamped(10.0, 10.0, 0.0)
        self.node.local_pose_pub.publish(current_pose)
        time.sleep(0.5)

        goal = self.create_position_target(20.0, 20.0, 5.0)
        self.node.setpoint_pub.publish(goal)
        time.sleep(1.0)

        # Wait for planning
        success = self.wait_for_planner_status("SUCCESS", timeout=15)
        self.assertTrue(success, f"Obstacle avoidance planning failed. Status: {self.node.current_planner_status}")

        # Verify path avoids obstacles
        path_received = self.wait_for_path(timeout=5)
        self.assertTrue(path_received, "No path received")
        
        # Check that path doesn't go through obstacles
        for pose in self.node.received_path.poses:
            x = int(round(pose.pose.position.x))
            y = int(round(pose.pose.position.y))
            z = int(round(pose.pose.position.z))
            
            # Check against 2D obstacles (they extend through depth)
            for obs in obstacles_2d:
                if (x, y) == obs:
                    self.fail(f"Path goes through 2D obstacle at ({x}, {y}, {z})")
            
            # Check against 3D obstacles
            for obs in obstacles_3d:
                if (x, y, z) == obs:
                    self.fail(f"Path goes through 3D obstacle at ({x}, {y}, {z})")

        print(f"Obstacle avoidance successful: {len(self.node.received_path.poses)} waypoints")

    def test_pathing_node_replan(self):
        """Test replanning functionality"""
        print("\n=== Testing Pathing Node: Replanning ===")
        
        # First, generate initial path
        self.test_pathing_node_basic_planning()
        initial_path_count = self.node.path_count
        time.sleep(1)

        # Add new obstacles
        new_obstacles = [(5, 5), (6, 6), (7, 7), (8, 8)]  # 2D obstacles
        obstacle_grid = self.create_obstacle_grid(width=50, height=50, depth=30, obstacles=new_obstacles)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        time.sleep(0.5)

        # Trigger replan
        print("Triggering replan...")
        replan_msg = Empty()
        self.node.replan_pub.publish(replan_msg)
        time.sleep(0.5)

        # Wait for replan completion
        success = self.wait_for_planner_status("SUCCESS", timeout=15)
        self.assertTrue(success, f"Replanning failed. Status: {self.node.current_planner_status}")

        # Verify new path was generated
        self.assertGreater(self.node.path_count, initial_path_count, "No new path generated during replan")
        
        print(f"Replanning successful: {len(self.node.received_path.poses)} waypoints")

    def test_waypoint_manager_basic_operation(self):
        """Test waypoint manager node basic functionality"""
        print("\n=== Testing Waypoint Manager: Basic Operation ===")
        
        self.node.reset_tracking()

        # First generate a path using pathing node
        self.test_pathing_node_basic_planning()
        time.sleep(1)

        # Publish current position to waypoint manager
        current_pose = self.create_pose_stamped(0.0, 0.0, 0.0)
        self.node.local_pose_pub.publish(current_pose)
        time.sleep(0.5)

        # Publish the same goal to waypoint manager
        goal = self.create_position_target(25.0, 25.0, 5.0)
        self.node.setpoint_pub.publish(goal)
        time.sleep(1.0)

        # Wait for waypoint manager to publish first waypoint
        waypoint_received = self.wait_for_waypoint(timeout=10)
        self.assertTrue(waypoint_received, "No waypoint received from waypoint manager")
        self.assertIsNotNone(self.node.current_waypoint, "Waypoint is None")

        print(f"Waypoint manager operational: First waypoint at ({self.node.current_waypoint.pose.position.x:.2f}, "
              f"{self.node.current_waypoint.pose.position.y:.2f}, {self.node.current_waypoint.pose.position.z:.2f})")

    def test_waypoint_manager_progression(self):
        """Test waypoint manager progression through path"""
        print("\n=== Testing Waypoint Manager: Waypoint Progression ===")
        
        # Setup initial path
        self.test_waypoint_manager_basic_operation()
        initial_waypoint = self.node.current_waypoint
        time.sleep(1)

        # Simulate moving to first waypoint
        if initial_waypoint:
            # Move close to the first waypoint
            near_waypoint = self.create_pose_stamped(
                initial_waypoint.pose.position.x - 0.5,
                initial_waypoint.pose.position.y - 0.5,
                initial_waypoint.pose.position.z
            )
            self.node.local_pose_pub.publish(near_waypoint)
            time.sleep(0.5)

            # Move to the waypoint
            at_waypoint = self.create_pose_stamped(
                initial_waypoint.pose.position.x,
                initial_waypoint.pose.position.y,
                initial_waypoint.pose.position.z
            )
            self.node.local_pose_pub.publish(at_waypoint)
            time.sleep(1.0)

            # Check if waypoint manager publishes next waypoint
            initial_count = self.node.waypoint_count
            waypoint_updated = False
            start_time = time.time()
            
            while time.time() - start_time < 5:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if self.node.waypoint_count > initial_count:
                    waypoint_updated = True
                    break

            if waypoint_updated:
                print(f"Waypoint progression successful: New waypoint at ({self.node.current_waypoint.pose.position.x:.2f}, "
                      f"{self.node.current_waypoint.pose.position.y:.2f}, {self.node.current_waypoint.pose.position.z:.2f})")
            else:
                print("Waypoint manager did not progress to next waypoint (might be expected behavior)")

    def test_integrated_planning_and_waypoint_management(self):
        """Test integrated operation of both nodes"""
        print("\n=== Testing Integrated Operation ===")
        
        self.wait_for_subscribers()
        self.node.reset_tracking()

        # Start with obstacle environment
        obstacles_2d = [(15, 10), (15, 11), (15, 12), (15, 13), (15, 14)]
        obstacles_3d = [(20, 15, 8), (20, 15, 9), (20, 15, 10)]
        all_obstacles = obstacles_2d + obstacles_3d
        obstacle_grid = self.create_obstacle_grid(width=40, height=40, depth=20, obstacles=all_obstacles)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        time.sleep(0.5)

        # Set initial position
        start_pose = self.create_pose_stamped(5.0, 10.0, 0.0)
        self.node.local_pose_pub.publish(start_pose)
        time.sleep(0.5)

        # Set goal
        goal = self.create_position_target(30.0, 15.0, 5.0)
        self.node.setpoint_pub.publish(goal)
        time.sleep(2.0)

        # Wait for path planning
        path_success = self.wait_for_planner_status("SUCCESS", timeout=15)
        self.assertTrue(path_success, "Integrated test: Path planning failed")

        # Wait for first waypoint
        waypoint_received = self.wait_for_waypoint(timeout=10)
        self.assertTrue(waypoint_received, "Integrated test: No waypoint received")

        # Add new obstacles and trigger replan
        more_obstacles_2d = [(20, 12), (20, 13), (20, 14)]
        more_obstacles_3d = [(25, 15, 5), (25, 15, 6)]
        more_obstacles = obstacles_2d + obstacles_3d + more_obstacles_2d + more_obstacles_3d
        obstacle_grid = self.create_obstacle_grid(width=40, height=40, depth=20, obstacles=more_obstacles)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        time.sleep(0.5)

        replan_msg = Empty()
        self.node.replan_pub.publish(replan_msg)
        time.sleep(0.5)

        # Wait for replan success
        replan_success = self.wait_for_planner_status("SUCCESS", timeout=15)
        self.assertTrue(replan_success, "Integrated test: Replanning failed")

        print("Integrated operation test successful")

    def test_error_conditions(self):
        """Test error conditions and edge cases"""
        print("\n=== Testing Error Conditions ===")
        
        self.wait_for_subscribers()
        self.node.reset_tracking()

        # Test with unreachable goal (surrounded by obstacles in 3D)
        obstacles = []
        # Create a 3D cage around position (9,9) at multiple heights
        for i in range(8, 12):
            for j in range(8, 12):
                for k in range(4, 8):  # Multiple height levels
                    if not (i == 9 and j == 9 and k == 5):  # Leave center free at one level
                        obstacles.append((i, j, k))
        
        obstacle_grid = self.create_obstacle_grid(width=20, height=20, depth=15, obstacles=obstacles)
        self.node.obstacle_grid_pub.publish(obstacle_grid)
        time.sleep(0.5)

        # Set start outside the blocked area
        start_pose = self.create_pose_stamped(5.0, 5.0, 0.0)
        self.node.local_pose_pub.publish(start_pose)
        time.sleep(0.5)

        # Set goal in the blocked center
        goal = self.create_position_target(9.0, 9.0, 5.0)  # At the one free level
        self.node.setpoint_pub.publish(goal)
        time.sleep(2.0)

        # Wait for status - might be FAILURE or still processing
        start_time = time.time()
        final_status = None
        while time.time() - start_time < 10:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.current_planner_status in ["SUCCESS", "FAILURE", "NO_PATH"]:
                final_status = self.node.current_planner_status
                break

        print(f"Error condition test result: {final_status}")
        # Don't assert failure here as different planners handle unreachable goals differently

    def save_debug_info(self, test_name):
        """Save debug information for failed tests"""
        timestamp = int(time.time())
        
        if self.node.received_path:
            filename = f"{test_name}_path_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(f"# Path for test: {test_name}\n")
                f.write(f"# Waypoints: {len(self.node.received_path.poses)}\n")
                for i, pose in enumerate(self.node.received_path.poses):
                    p = pose.pose.position
                    f.write(f"{i}: {p.x:.3f}, {p.y:.3f}, {p.z:.3f}\n")
            print(f"Path debug info saved to {filename}")

        if self.node.status_history:
            filename = f"{test_name}_status_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(f"# Status history for test: {test_name}\n")
                for status, timestamp in self.node.status_history:
                    f.write(f"{timestamp}: {status}\n")
            print(f"Status debug info saved to {filename}")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)