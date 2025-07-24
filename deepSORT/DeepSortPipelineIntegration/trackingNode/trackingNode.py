import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from kalman_filter import KalmanFilter
from cv_bridge import CVBridge
from vision_msgs.msg import Detection2DArray
from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
from custom_msgs.msg import TrackArray, Track
import cv2 
import json 


class TrackingNode(Node):
    def __init__(self):
            super().__init__('tracking_node')
            self.sub_det = self.create_subscription(Detection2DArray, '/kestrel/detections', self.dets_cb, 10)
            self.pub_tracks = self.create_publisher(TrackArray, '/kestrel/tracks', 10)
            # Initialize Kalman filter class here

    def dets_cb(self, det_msg):
        # 1. Predict all tracks call predict_tracks
        # //extract appearance features from the cnn
        # 2. Gate & associate (week-2)
        # 3. Update or create new tracks call update_tracks()
        self.pub_tracks.publish(track_array_msg)
    def predict_tracks(self):
        #loop through current tracks and call Kalman prediction on each one
        #update all tracks
        '''
            For matched: use kalman.correct() and update state.
            For unmatched_tracks: increase age or missed_count.
            For unmatched_detections: create new track with Kalman + appearance.
        '''
 

#Kalman Filter

#function must have prediction phase and correction phase update phase)

#mahalanobis gating

#extracting appearance features from CNN