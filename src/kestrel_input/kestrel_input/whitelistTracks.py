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
        
        # Short history of previous images, with timestamp as key
        self.images = {}
        self.image_buffer_size = 30 # decide max size

        # Will be used to store the raw image linked to current batch of tracks (using timestamp)
        # we can then crop this image based on coordinates from each track
        self.image_to_display = None

        self.bridge = CvBridge()
        self.selected_tracks = set() # Used to know which tracks the user has selected to follow
        self.displayed_tracks = set() # Keep track of which tracks are displayed in gui

        # Initalize ros publishers/subscribers
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.tracking_subscriber = self.create_subscription(TrackArray, '/kestrel/tracks', self.tracking_callback, 10)
        self.pub_input = self.create_publisher(TrackArray, '/kestrel/whitelist_tracks', 10)

    def image_callback(self, msg: Image):
        # convert to nano sec
        timestamp = msg.header.stamp.sec * 1e9 + msg.header.nanosec
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        self.images[timestamp] = frame

        # Shrink the size of images stored
        if len(self.images) > self.image_buffer_size:
            oldest_frame = min(self.images.keys())
            del self.images[oldest_frame]


    def tracking_callback(self, msg):
        timestamp = msg.header.stamp.sec * 1e9 + msg.header.nanosec

        # current set of tracks was linked to this timestamp
        if timestamp in self.images:
            # use this image to associate with this set of tracks
            self.image_to_display = self.images[timestamp]

            self.tracks = msg.tracks


    # handles loop for gui
    def run_gui(self):
        # maybe make local copies of class variables from above for overriding reasons (image_to_display, tracks)
        # new tracks/raw image override existing data, so we need to stop race conditions

        if self.image_to_display:
            # prepare image to get cropped if needed
            # check if self.displayed_tracks has some tracks that no longer exist
            # crop bboxs from self.image_to_display and display them
            # handle user clicks 
            pass

        # probably change this depending on what we do above, but this is the main idea
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