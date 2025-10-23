#!/usr/bin/env python3
import unittest
import time
import subprocess
import sys
import threading
import pytest

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Empty
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
        try:
            cls.node_process.wait(timeout=5)
        except Exception:
            cls.node_process.kill()
        rclpy.shutdown()

    # ----------------- Helpers -----------------
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

    def wait_for_next_path(self, prev_count=None, timeout=8):
        if prev_count is None:
            prev_count = self.node.path_count
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.path_count > prev_count:
                return True
        return False

    def assert_path_avoids(self, path_msg, obstacle_coords):
        occ = set(obstacle_coords or [])
        for ps in path_msg.poses:
            x = int(round(ps.pose.position.x))
            y = int(round(ps.pose.position.y))
            self.assertFalse((x, y) in occ, f"Path intersects obstacle cell {(x,y)}")

    def time_to_next_path(self, trigger, timeout=10):
        prev = self.node.path_count
        t0 = time.perf_counter()
        trigger()
        self.assertTrue(self.wait_for_next_path(prev, timeout=timeout), "Timed out waiting for path")
        t1 = time.perf_counter()
        return (t1 - t0), self.node.received_path

    # ----------------- Tests -----------------
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

        self.replan_pub.publish(Empty())
        time.sleep(0.5)

        success = self.wait_for_status("SUCCESS", timeout=10)
        self.assertTrue(success, f"Planner did not reach SUCCESS status. Current: {self.node.current_status}")

        path_received = self.wait_for_path(timeout=5)
        self.assertTrue(path_received, "No path received from planner")
        self.assertIsNotNone(self.node.received_path, "Path is None")
        self.assertGreater(len(self.node.received_path.poses), 0, "Path is empty")
        self.assert_path_avoids(self.node.received_path, obstacle_coords=[(1, 1)])
        print(f"Initial planning successful: {len(self.node.received_path.poses)} waypoints")

    def test_replan_with_new_obstacles(self):
        print("\n=== Testing Replan with New Obstacles ===")
        self.test_initial_path_planning()

        initial_path_count = self.node.path_count
        time.sleep(1)

        obstacles = [(1, 1), (25, 2), (25, 3), (25, 4), (25, 5)]
        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=obstacles))
        time.sleep(0.5)

        print("Triggering replan...")
        self.replan_pub.publish(Empty())
        time.sleep(0.5)

        start_time = time.time()
        while time.time() - start_time < 10:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.current_status == "REPLANNING":
                print("Replanning status detected")
                break
            elif self.node.current_status == "SUCCESS" and self.node.path_count > initial_path_count:
                print("Replan completed directly to SUCCESS")
                break

        success = self.wait_for_status("SUCCESS", timeout=15)
        self.assertTrue(success, f"Replanning did not complete successfully. Status: {self.node.current_status}")
        self.assertGreater(self.node.path_count, initial_path_count, "No new path generated during replan")
        self.assert_path_avoids(self.node.received_path, obstacle_coords=obstacles)
        print(f"Replan successful: {len(self.node.received_path.poses)} waypoints")

    def test_replan_without_existing_path(self):
        print("\n=== Testing Replan without Existing Path ===")
        self.wait_for_subscribers()
        self.node.reset_path_tracking()

        empty_map = self.create_map_with_obstacles()
        self.map_pub.publish(empty_map)
        time.sleep(0.5)

        self.publish_start_goal(start_pos=(10.0, 10.0, 0.0), goal_pos=(80.0, 80.0, 0.0))
        time.sleep(0.5)

        print("Triggering replan without existing path...")
        self.replan_pub.publish(Empty())
        time.sleep(0.5)

        success = self.wait_for_status("SUCCESS", timeout=10)
        self.assertTrue(success, f"Replan without existing path failed. Status: {self.node.current_status}")

        path_received = self.wait_for_path(timeout=5)
        self.assertTrue(path_received, "No path received from replan command")
        print(f"Replan without existing path successful: {len(self.node.received_path.poses)} waypoints")

    def test_replan_with_goal_change(self):
        print("\n=== Testing Replan with Goal Change ===")
        self.test_initial_path_planning()
        initial_path_count = self.node.path_count
        time.sleep(1)

        print("Changing goal position...")
        self.publish_start_goal(goal_pos=(20.0, 80.0, 20.0))
        time.sleep(0.5)

        print("Triggering replan with new goal...")
        self.replan_pub.publish(Empty())
        time.sleep(0.5)

        success = self.wait_for_status("SUCCESS", timeout=10)
        self.assertTrue(success, f"Goal change replan failed. Status: {self.node.current_status}")
        self.assertGreater(self.node.path_count, initial_path_count, "No new path after goal change")
        print(f"Goal change replan successful: {len(self.node.received_path.poses)} waypoints")

    def test_multiple_replans(self):
        print("\n=== Testing Multiple Consecutive Replans ===")
        self.test_initial_path_planning()
        initial_count = self.node.path_count

        for i in range(3):
            print(f"Performing replan #{i+1}")
            obstacles = [(10 + i * 5, 10 + i * 5), (15 + i * 5, 15 + i * 5)]
            self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=obstacles))
            time.sleep(0.3)

            self.replan_pub.publish(Empty())
            time.sleep(0.3)

            success = self.wait_for_status("SUCCESS", timeout=8)
            self.assertTrue(success, f"Multiple replan #{i+1} failed. Status: {self.node.current_status}")

        final_count = self.node.path_count
        self.assertGreaterEqual(final_count - initial_count, 3,
                                f"Expected at least 3 new paths, got {final_count - initial_count}")
        print(f"Multiple replans successful: {final_count - initial_count} new paths generated")

    # ----------------- Auto-replan tests (no /replan publish) -----------------
    def test_auto_replan_move_goal(self):
        self.test_initial_path_planning()
        base_obstacles = [(1, 1)]
        prev = self.node.path_count

        self.publish_start_goal(goal_pos=(20.0, 80.0, 0.0))
        self.assertTrue(self.wait_for_next_path(prev, timeout=10), "Auto-replan not triggered by goal change")
        self.assertIsNotNone(self.node.received_path)
        self.assert_path_avoids(self.node.received_path, base_obstacles)

    def test_auto_replan_move_goal_and_start(self):
        self.test_initial_path_planning()
        base_obstacles = [(1, 1)]
        prev = self.node.path_count

        self.publish_start_goal(start_pos=(10.0, 10.0, 0.0), goal_pos=(80.0, 20.0, 0.0))
        self.assertTrue(self.wait_for_next_path(prev, timeout=10), "Auto-replan not triggered by start+goal change")
        p = self.node.received_path
        my_path = p  # alias for clarity
        self.assertIsNotNone(my_path)
        self.assert_path_avoids(my_path, base_obstacles)

    def test_auto_replan_move_obstacles(self):
        self.wait_for_subscribers()
        self.node.reset_path_tracking()
        base_obstacles = []
        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=base_obstacles))
        time.sleep(0.3)
        self.publish_start_goal(start_pos=(2.0, 2.0, 0.0), goal_pos=(90.0, 90.0, 0.0))
        self.assertTrue(self.wait_for_next_path(timeout=10), "No initial path produced")
        prev = self.node.path_count

        added = [(x, 25) for x in range(3, 95)]
        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=added))

        self.assertTrue(self.wait_for_next_path(prev, timeout=10), "Auto-replan not triggered by map change")
        p = self.node.received_path
        self.assertIsNotNone(p)
        self.assert_path_avoids(p, added)

    def test_auto_replan_move_goal_start_and_obstacles(self):
        self.wait_for_subscribers()
        self.node.reset_path_tracking()
        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=[]))
        time.sleep(0.3)
        self.publish_start_goal(start_pos=(5.0, 25.0, 0.0), goal_pos=(95.0, 25.0, 0.0))
        self.assertTrue(self.wait_for_next_path(timeout=10), "No initial path produced")
        prev = self.node.path_count

        new_start = (10.0, 10.0, 0.0)
        new_goal = (90.0, 40.0, 0.0)
        block_corridor = [(x, 25) for x in range(8, 92)]
        shift_remove = [(10, y) for y in range(20, 30)]
        shift_add = [(15, y) for y in range(20, 30)]
        new_obstacles = list((set(block_corridor) - set(shift_remove)) | set(shift_add))

        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=new_obstacles))
        time.sleep(0.2)
        self.publish_start_goal(start_pos=new_start, goal_pos=new_goal)

        self.assertTrue(self.wait_for_next_path(prev, timeout=10), "Auto-replan not triggered by combined changes")
        p = self.node.received_path
        self.assertIsNotNone(p)
        self.assert_path_avoids(p, new_obstacles)

    # ----------------- Metrics -----------------
    def test_metrics_plan_vs_replan_and_obstacle_free(self):
        self.wait_for_subscribers()
        self.node.reset_path_tracking()

        base_obstacles = []
        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=base_obstacles))
        time.sleep(0.2)

        start_xy = (2.0, 2.0, 0.0)
        goal_xy = (90.0, 90.0, 0.0)

        plan_t, path0 = self.time_to_next_path(
            trigger=lambda: self.publish_start_goal(start_pos=start_xy, goal_pos=goal_xy),
            timeout=12
        )
        self.assertIsNotNone(path0)
        self.assert_path_avoids(path0, base_obstacles)

        added = [(50, y) for y in range(40, 60)]
        self.map_pub.publish(self.create_map_with_obstacles(obstacle_coords=added))
        time.sleep(0.1)

        replan_t, path1 = self.time_to_next_path(trigger=lambda: None, timeout=12)
        self.assertIsNotNone(path1)
        self.assert_path_avoids(path1, added)

        self.assertLessEqual(
            replan_t, max(3.0 * plan_t, 0.300),
            f"Replan too slow: replan={replan_t:.3f}s, plan={plan_t:.3f}s"
        )
        print(f"[METRICS] initial_plan={plan_t:.4f}s replan={replan_t:.4f}s")

    # ----------------- Debug helper -----------------
    def save_path_to_file(self, filename="test_path.txt"):
        if self.node.received_path:
            with open(filename, "w") as f:
                f.write(f"# Path with {len(self.node.received_path.poses)} waypoints\n")
                for i, pose in enumerate(self.node.received_path.poses):
                    p = pose.pose.position
                    f.write(f"{i}: {p.x:.3f}, {p.y:.3f}, {p.z:.3f}\n")
            print(f"Path saved to {filename}")

# --- ensure pytest discovers unittest.TestCase tests ---
def load_tests(loader, tests, pattern):
    return loader.loadTestsFromTestCase(TestPlanner)
# --- pytest shim: guarantee pytest runs our unittest suite ---
import pytest

@pytest.mark.unittest
def test_run_unittest_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPlanner)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
if __name__ == '__main__':
    unittest.main(verbosity=2)
