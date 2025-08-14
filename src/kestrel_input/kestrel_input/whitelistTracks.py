import rclpy
from rclpy.node import Node
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from sensor_msgs.msg import Image
from kestrel_msgs.msg import TrackArray

from cv_bridge import CvBridge


class InputNode(Node):
    """
    This node subscribes to image and tracking output. Displays tracks through a gui, to let
    user track certain people and publishes selections to clustering node
    """
    def __init__(self):
        super().__init__('input_node')
        self.tracks = []
        """
        I think we need to keep track of multiple frames bc of latency. We match tracks and detections
        at frame 10, but camera may now be at frame 20. So we need to go to image from frame 10 to get
        the bboxes from there
        """
        self.latest_frame = None

        self.bridge = CvBridge()
        self.selected_tracks = set() # Used to know which tracks the user has selected to follow
        self.displayed_tracks = set() # Keep track of which tracks are displayed in gui

        # Initalize ros publishers/subscribers
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.tracking_subscriber = self.create_subscription(TrackArray, '/kestrel/tracks', self.tracking_callback, 10)
        self.pub_input = self.create_publisher(TrackArray, '/kestrel/whitelist_tracks', 10)

    def image_callback(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def tracking_callback(self, msg):
        self.tracks = msg.tracks

    # handles loop for gui
    def run_gui(self):
        """
        1. get track ids from self.tracks
        2. Remove tracks that no longer exist, but are still displayed (self.displayed_tracks)
        3. Display current tracks
           - use timestamps to match image and tracks. Then we can get the corresponding image and crop
           it based on the bbox coordinates from tracking node (x1, y1, x2, y2)
        """

        if self.selected_tracks:
            selected = []
            for track in self.tracks:
                if track.id in self.selected_tracks:
                    selected.append(track)

            whitelisted_tracks = TrackArray()
            # whitelisted_tracks.header = None idk what this should be
            whitelisted_tracks.tracks = selected
            self.pub_input(whitelisted_tracks)

'''
def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
'''