#!/usr/bin/env python3
import unittest
import time
import subprocess
import sys
import threading

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Empty


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
        self.received_path = msg

    def status_callback(self, msg):
        self.current_status = msg.data
        self.status_history.append((msg.data, time.time()))

    def reset_path_tracking(self):
        self.received_path = None
        self.path_count = 0
        self.last_path_time = None
        self.status_history = []


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # launch the planner node executable
        cls.node_process = subprocess.Popen(
            ["ros2", "run", "kestrel_planning", "kestrel_planning"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def _pipe(stream, prefix):
            for line in iter(stream.readline, ''):
                sys.stdout.write(f"[{prefix}] {line}")
            stream.close()

        cls._stdout_thread = threading.Thread(target=_pipe, args=(cls.node_process.stdout, "STDOUT"), daemon=True)
        cls._stderr_thread = threading.Thread(target=_pipe, args=(cls.node_process.stderr, "STDERR"), daemon=True)
        cls._stdout_thread.start()
        cls._stderr_thread.start()

        time.sleep(2)  # allow node to start

        rclpy.init()
        cls.node = PlannerTestNode()

        # pubs used by tests
        cls.map_pub = cls.node.create_publisher(OccupancyGrid, '/map', 10)
        cls.start_pub = cls.node.create_publisher(PoseStamped, '/current_pose', 10)
        cls.goal_pub = cls.node.create_publisher(PoseStamped, '/goal
