import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from filterpy.kalman import KalmanFilter as FP_KalmanFilter
from cv_bridge import CVBridge
from vision_msgs.msg import Detection2DArray
from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
from custom_msgs.msg import TrackArray, Track
from typing import List, Tuple
from numpy.linalg import inv
from scipy.stats import chi2
import cv2 
import json 


def det2bbox(det):
    """Extract [x1,y1,x2,y2] from a vision_msgs/Detection2D."""


def bbox2state(bbox):
    """Convert bbox → Kalman state vector. Convert [x1,y1,x2,y2] → [cx, cy, w, h] """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.array([cx, cy, w, h])

def mahalanobis_gate(kalmanFilter, det_state, covariance, thresh):
    """Return True if Mahalanobis distance < thresh."""
    #Replaced track_State with kalmanFilter
    #What we are passing as covariance will be the innovation covariance which is S (S = H*P*H^T + R), should be calculate in measurement matrix, if not will need to do here
    
    # I can just implement it in here *so its subject to change but for now ima add it as a parameter, same thing with S I think it should be calculated in measurement matrix

    #Convert raw bbox into z (actual measurement)
    z= bbox2state(det_state).reshape(4,1)
    
    H = kalmanFilter.H
    x = kalmanFilter.x
    
    z_hat = H @ x 
    y = z - z_hat
    #Due to gating taking the squared Mahalanobis Distance is considered better and standard practice then using the normal Distance
    distance_squared = float(y.T @ np.linalg.inv(covariance) @ y)
    

    return distance_squared < thresh

def build_cost_matrix(tracks, detections, w_motion, w_app):
    """Combine motion & appearance costs into an (N×M) matrix."""
    N = len(tracks)
    M = len(detections)
    cost = np.zeros((N, M))
    # TODO: fill cost[i,j] = w_motion*motion_cost + w_app*appearance_cost
    return cost

def assign_detections(cost_matrix):
    """
    Solve Hungarian:
    returns matches, unmatched_tracks, unmatched_dets
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # …

class Track:
    def __init__(self, track_id: int, initial_bbox: List[int], dt: float = 1.0):
        self.id = track_id
        # instantiate FilterPy KalmanFilter
        self.kf = FP_KalmanFilter(dim_x=8, dim_z=4)
        self._init_kf(dt)
        # initialize state
        

    def _init_kf(self, dt: float):
        # State transition (constant velocity)
        F = np.eye(8)
        for i in range(4):
            F[i, i+4] = dt
        self.kf.F = F
        # Measurement matrix (observe cx,cy,w,h)
        

    def predict(self) -> List[int]:
        self.kf.predict()
        # Convert state → bbox
        

    def update(self, detection_bbox: List[int], feature: np.ndarray) -> List[int]:
        z = bbox2state(detection_bbox).reshape(4,1)
        self.kf.update(z)
        cx, cy, w, h = self.kf.x[:4,0]
        x1, y1 = cx - w/2, cy - h/2
        self.bbox = [int(x1), int(y1), int(x1+w), int(y1+h)]
        self.feature = feature
        self.missed = 0
        return self.bbox

class TrackManager:
    def __init__(self,max_age: int,min_hits: int,gate_thresh: float,w_motion: float,w_app: float):
        self.tracks: List[Track] = []
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate_thresh = gate_thresh
        self.w_motion = w_motion
        self.w_app = w_app

    def predict_all(self) -> None:
        for trk in self.tracks:
            trk.predict()

    def update(self,detections: List[List[int]],features: List[np.ndarray]) -> List[Track]:
        # 1) Predict
        self.predict_all()

        # 2) Build cost matrix & gate

        # 3) Associate

        # 4) Update matched

        # 5) Handle unmatched tracks

        # 6) Create new tracks for unmatched detections

        # 7) Filter by min_hits (optional)

#ROS NODE
class TrackingNode(Node):
    def __init__(self):
            super().__init__('tracking_node')
            self.bridge = CVBridge()
            self.yolo = YOLO('yolov8n.pt')
            #initiate track manager
            self.sub_det = self.create_subscription(Detection2DArray, '/kestrel/detections', self.dets_cb, 10)
            self.pub_tracks = self.create_publisher(TrackArray, '/kestrel/tracks', 10)
            # Initialize Kalman filter class here
            self.kf = FP_KalmanFilter(dim_x=8, dim_z=4)

    def dets_cb(self, det_msg:Detection2DArray):
        # 1. Predict all tracks 
        # //extract appearance features from the cnn
        # 2. Gate & associate (week-2) (Extract bboxes)
        # 3. Update tracks or create new tracks 
        #4 Build and Publish
        self.pub_tracks.publish(track_array_msg)

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
#goals 

#Kalman Filter

#function must have prediction phase and correction phase update phase)

#mahalanobis gating

#extracting appearance features from CNN