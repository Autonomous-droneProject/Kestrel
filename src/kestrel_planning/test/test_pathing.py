import unittest
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Empty
import subprocess
import sys
import threading
from builtin_interfaces.msg import Time




def set_header(msg, frame_id="map"):
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(sec=int(time.time()))
    return msg

class PlannerTestNode(Node):
    def __init__(self):
        super().__init__('planner_test_node')

        self.received_path = None
        self.current_status = None
        self.status_history = []
        self.path_count = 0
        self.last_path_time = None

        self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.create_subscription(String, '/planner_status', self.status_callback, 10)

    def path_callback(self, msg):
        self.path_count += 1
        self.last_path_time = time.time()
        self.get_logger().info(f"Received path #{self.path_count} with {len(msg.poses)} waypoints")
        self.received_path = msg

    def status_callback(self, msg):
        self.current_status = msg.data
        self.status_history.append((msg.data, time.time()))
        print(f"[Planner Status] {msg.data}")
    
    def reset_path_tracking(self):
        """Reset path tracking for new test scenarios"""
        self.received_path = None
        self.path_count = 0
        self.last_path_time = None
        self.status_history = []


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node_process = subprocess.Popen(
            ["ros2", "run", "kestrel_planning", "kestrel_planning"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def print_stream(stream, prefix):
            for line in iter(stream.readline, ''):
                sys.stdout.write(f"[{prefix}] {line}")
            stream.close()

        cls.stdout_thread = threading.Thread(
            target=print_stream, args=(cls.node_process.stdout, "STDOUT"), daemon=True
        )
        cls.stderr_thread = threading.Thread(
            target=print_stream, args=(cls.node_process.stderr, "STDERR"), daemon=True
        )
        cls.stdout_thread.start()
        cls.stderr_thread.start()

        time.sleep(2)

        rclpy.init()
        cls.node = PlannerTestNode()

        cls.map_pub = cls.node.create_publisher(OccupancyGrid, '/map', 10)
        cls.start_pub = cls.node.create_publisher(PoseStamped, '/current_pose', 10)
        cls.goal_pub = cls.node.create_publisher(PoseStamped, '/goal_pose', 10)
        cls.replan_pub = cls.node.create_publisher(Empty, '/replan', 10)

    @classmethod
    def tearDownClass(cls):
        cls.node_process.terminate()
        cls.node_process.wait(timeout=5)
        rclpy.shutdown()
    
    def create_map_with_obstacles(self, width=100, height=100, obstacle_coords=None):
        map_msg = OccupancyGrid()
        map_msg.info.width = width
        map_msg.info.height = height
        map_msg.info.resolution = 1.0
        map_msg.data = [0] * (width * height) 

        if obstacle_coords:
            for (ox, oy) in obstacle_coords:
                if 0 <= ox < width and 0 <= oy < height:
                    idx = oy * width + ox
                    map_msg.data[idx] = 100 

        map_msg = set_header(map_msg, "map")
        return map_msg

    def publish_start_goal(self, start_pos=(0.0, 0.0, 0.0), goal_pos=(50.0, 5.0, 50.0)):
        start_msg = PoseStamped()
        start_msg.pose.position.x = start_pos[0]
        start_msg.pose.position.y = start_pos[1]
        start_msg.pose.position.z = start_pos[2]
        start_msg = set_header(start_msg, "map")
        self.start_pub.publish(start_msg)

        goal_msg = PoseStamped()
        goal_msg.pose.position.x = goal_pos[0]
        goal_msg.pose.position.y = goal_pos[1]
        goal_msg.pose.position.z = goal_pos[2]
        goal_msg = set_header(goal_msg, "map")
        self.goal_pub.publish(goal_msg)

    def wait_for_subscribers(self, timeout=5):
        start_time = time.time()
        while (self.map_pub.get_subscription_count() == 0 or
               self.start_pub.get_subscription_count() == 0 or
               self.goal_pub.get_subscription_count() == 0) and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def wait_for_status(self, expected_status, timeout=10):
        start_time = time.time()
        while self.node.current_status != expected_status and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.node.current_status == expected_status

    def wait_for_path(self, timeout=5):
        initial_count = self.node.path_count
        start_time = time.time()
        while self.node.path_count <= initial_count and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.node.path_count > initial_count

    def test_initial_path_planning(self):
        print("\n=== Testing Initial Path Planning ===")
        
        self.wait_for_subscribers()
        self.node.reset_path_tracking()

        empty_map = self.create_map_with_obstacles()
        self.map_pub.publish(empty_map)
        time.sleep(0.5)

        self.publish_start_goal()
        time.sleep(0.5)

        map_with_obstacle = self.create_map_with_obstacles(obstacle_coords=[(1, 1)])
        self.map_pub.publish(map_with_obstacle)
        time.sleep(0.5)

        replan_msg = Empty()
        self.replan_pub.publish(replan_msg)
        time.sleep(0.5)


        success = self.wait_for_status("SUCCESS", timeout=10)
        self.assertTrue(success, f"Planner did not reach SUCCESS status. Current: {self.node.current_status}")

        path_received = self.wait_for_path(timeout=5)
        self.assertTrue(path_received, "No path received from planner")
        self.assertIsNotNone(self.node.received_path, "Path is None")
        self.assertGreater(len(self.node.received_path.poses), 0, "Path is empty")
        
        for pose in self.node.received_path.poses:
            x = round(pose.pose.position.x)
            y = round(pose.pose.position.y)
            self.assertFalse((x, y) == (1, 1), f"Path includes blocked cell (1,1): ({x},{y})")

        print(f"Initial planning successful: {len(self.node.received_path.poses)} waypoints")

    def test_replan_with_new_obstacles(self):
        """Test replanning when new obstacles are added"""
        print("\n=== Testing Replan with New Obstacles ===")
        
        self.test_initial_path_planning()
        
        initial_path_count = self.node.path_count
        time.sleep(1)

        map_with_more_obstacles = self.create_map_with_obstacles(
            obstacle_coords=[(1, 1), (25, 2), (25, 3), (25, 4), (25, 5)]
        )
        self.map_pub.publish(map_with_more_obstacles)
        time.sleep(0.5)

        print("Triggering replan...")
        replan_msg = Empty()
        self.replan_pub.publish(replan_msg)
        time.sleep(0.5)

        # Wait for replanning status or success
        replanning_detected = False
        start_time = time.time()
        while time.time() - start_time < 10:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.current_status == "REPLANNING":
                replanning_detected = True
                print("Replanning status detected")
                break
            elif self.node.current_status == "SUCCESS" and self.node.path_count > initial_path_count:
                print("Replan completed directly to SUCCESS")
                break

        # Wait for final success
        success = self.wait_for_status("SUCCESS", timeout=15)
        self.assertTrue(success, f"Replanning did not complete successfully. Status: {self.node.current_status}")

        # Verify we got a new path
        self.assertGreater(self.node.path_count, initial_path_count, "No new path generated during replan")
        
        print(f"Replan successful: {len(self.node.received_path.poses)} waypoints")

    def test_replan_without_existing_path(self):
        """Test replan command when no existing path exists"""
        print("\n=== Testing Replan without Existing Path ===")
        
        self.wait_for_subscribers()
        self.node.reset_path_tracking()

        # Publish map and positions
        empty_map = self.create_map_with_obstacles()
        self.map_pub.publish(empty_map)
        time.sleep(0.5)

        self.publish_start_goal(start_pos=(10.0, 10.0, 0.0), goal_pos=(80.0, 80.0, 0.0))
        time.sleep(0.5)

        # Trigger replan before any initial planning
        print("Triggering replan without existing path...")
        replan_msg = Empty()
        self.replan_pub.publish(replan_msg)
        time.sleep(0.5)

        # Should result in new path planning
        success = self.wait_for_status("SUCCESS", timeout=10)
        self.assertTrue(success, f"Replan without existing path failed. Status: {self.node.current_status}")

        path_received = self.wait_for_path(timeout=5)
        self.assertTrue(path_received, "No path received from replan command")
        
        print(f"Replan without existing path successful: {len(self.node.received_path.poses)} waypoints")

    def test_replan_with_goal_change(self):
        """Test replanning after goal change"""
        print("\n=== Testing Replan with Goal Change ===")
        
        # Start with initial planning
        self.test_initial_path_planning()
        initial_path_count = self.node.path_count
        time.sleep(1)

        # Change goal position
        print("Changing goal position...")
        self.publish_start_goal(goal_pos=(20.0, 80.0, 20.0))
        time.sleep(0.5)

        # Trigger replan
        print("Triggering replan with new goal...")
        replan_msg = Empty()
        self.replan_pub.publish(replan_msg)
        time.sleep(0.5)

        # Wait for completion
        success = self.wait_for_status("SUCCESS", timeout=10)
        self.assertTrue(success, f"Goal change replan failed. Status: {self.node.current_status}")

        # Verify new path
        self.assertGreater(self.node.path_count, initial_path_count, "No new path after goal change")
        
        print(f"Goal change replan successful: {len(self.node.received_path.poses)} waypoints")

    def test_multiple_replans(self):
        """Test multiple consecutive replans"""
        print("\n=== Testing Multiple Consecutive Replans ===")
        
        # Start with initial path
        self.test_initial_path_planning()
        initial_count = self.node.path_count
        
        # Perform multiple replans
        for i in range(3):
            print(f"Performing replan #{i+1}")
            
            # Add different obstacles each time
            obstacles = [(10+i*5, 10+i*5), (15+i*5, 15+i*5)]
            map_msg = self.create_map_with_obstacles(obstacle_coords=obstacles)
            self.map_pub.publish(map_msg)
            time.sleep(0.3)

            # Trigger replan
            replan_msg = Empty()
            self.replan_pub.publish(replan_msg)
            time.sleep(0.3)

            # Wait for success
            success = self.wait_for_status("SUCCESS", timeout=8)
            self.assertTrue(success, f"Multiple replan #{i+1} failed. Status: {self.node.current_status}")

        # Verify we got multiple new paths
        final_count = self.node.path_count
        self.assertGreaterEqual(final_count - initial_count, 3, 
                               f"Expected at least 3 new paths, got {final_count - initial_count}")
        
        print(f"Multiple replans successful: {final_count - initial_count} new paths generated")

    def save_path_to_file(self, filename="test_path.txt"):
        """Save the current path to a file for debugging"""
        if self.node.received_path:
            with open(filename, "w") as f:
                f.write(f"# Path with {len(self.node.received_path.poses)} waypoints\n")
                for i, pose in enumerate(self.node.received_path.poses):
                    p = pose.pose.position
                    f.write(f"{i}: {p.x:.3f}, {p.y:.3f}, {p.z:.3f}\n")
            print(f"Path saved to {filename}")


if __name__ == '__main__':
    # Run tests with more verbose output
    unittest.main(verbosity=2)