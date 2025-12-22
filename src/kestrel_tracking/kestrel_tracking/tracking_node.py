#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from filterpy.kalman import KalmanFilter as FP_KalmanFilter
from cv_bridge import CvBridge
from kestrel_msgs.msg import DetectionArray, Track as TrackMsg, TrackArray
from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Sequence
from numpy.linalg import inv
from scipy.stats import chi2
import cv2 
import json 

#Extracts bounding box from a vision_msgs/Detection message
#This function converts the center and size of the bounding box into the format [x1, y1, x2, y2]
#where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
def det2bbox(det): 
    """Extract [x1,y1,x2,y2] from a kestrel_msgs/msg/EmbeddedDetection2D."""
    # extract and return the center directly from th vision msgs
    return [int(det.x1), int(det.y1), int(det.x2), int(det.y2)] 

def bbox2state(bbox):
    """Convert bbox → Kalman state vector. Convert [x1,y1,x2,y2] → [cx, cy, w, h] """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.array([cx, cy, w, h])
def state2bbox(state: Sequence[float])-> List[int]:
    """
    Convert a Kalman state vector → [x1, y1, x2, y2].
    Expects state[:4] = [cx, cy, w, h].
    """
    cx, cy, w, h = state[:4]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [int(x1), int(y1), int(x2), int(y2)]

def mahalanobis_gate(kalmanFilter, det_bbox, innovation_covariance, thresh):
    """Return True if Mahalanobis distance < thresh."""
    #Replaced track_State with kalmanFilter
    #What we are passing as covariance will be the innovation covariance which is S (S = H*P*H^T + R), should be calculate in measurement matrix, if not will need to do here

    # I can just implement it in here *so its subject to change but for now ima add it as a parameter, same thing with S I think it should be calculated in measurement matrix

    #Convert raw bbox into z (actual measurement)
    z = bbox2state(det_bbox).reshape(4,1)

    H = kalmanFilter.H
    x = kalmanFilter.x

    z_hat = H @ x 
    y = z - z_hat
    
    # S*x = y is the same thing as the inverse of S (S is the innovation covariance and we neeed the inverse)
    temp = np.linalg.solve(innovation_covariance, y)
    
    #Due to gating taking the squared Mahalanobis Distance is considered better and standard practice then using the normal Distance
    distance_squared = y.t @ temp


    return distance_squared < thresh

def appearance_cost(track, det_feature):
    # cosine distance = 1 - cosine similarity
    if track.feature is None:
        return 1.0  # max cost if no feature
    f1 = track.feature / np.linalg.norm(track.feature)
    f2 = det_feature / np.linalg.norm(det_feature)
    return 1.0 - np.dot(f1, f2)

def build_cost_matrix(tracks, detections,features, w_motion, w_app):
    """Combine motion & appearance costs into an (N×M) matrix."""
    N = len(tracks)
    M = len(detections)
    cost = np.zeros((N, M))
    # TODO: fill cost[i,j] = w_motion*motion_cost + w_app*appearance_cost
    
    for i,track in enumerate(tracks):
            for j,det_bbox in enumerate(detections):
                mc = motion_cost(track,det_bbox)
                ac = appearance_cost(track,features[j])
                cost[i,j] = w_motion * mc + w_app * ac
    return cost

def assign_detections(cost_matrix):
    """
    Solve Hungarian:
    returns matches, unmatched_tracks, unmatched_dets
    """
    #1 run hungarian to get tje minimal cost assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # 2) Build the matched pairs
    matches = list(zip(row_ind.tolist(), col_ind.tolist()))

    # 3) Determine which tracks/detections were left out
    all_tracks = set(range(cost_matrix.shape[0]))
    all_dets   = set(range(cost_matrix.shape[1]))
    matched_tracks = set(row_ind.tolist())
    matched_detections = set(col_ind.tolist())

    unmatched_tracks = list(all_tracks - matched_tracks)
    unmatched_dets   = list(all_dets   - matched_detections)

    return matches, unmatched_tracks, unmatched_dets

class Track:
    def __init__(self, track_id: int, initial_bbox: List[int], dt: float = 1.0):
        self.id = track_id
        # instantiate FilterPy KalmanFilter
        self.kf = FP_KalmanFilter(dim_x=8, dim_z=4)
        self._init_kf(dt)
        # initialize state
        initial_state = bbox2state(initial_bbox)
        self.kf.x = np.array([
            initial_state[0],  # cx (Center X)
            initial_state[1],  # cy (Center Y
            initial_state[2],  # w (Width)
            initial_state[3],  #h (height)
            0,                 #vx (velocity in x)
            0,                 #vy (velocity in y)  
            0,                 #vw (velocity in w)
            0                  #vh (velocity in h)
            
        ]).reshape(8, 1)

        #Initialize other attributes
        self.bbox = initial_bbox  # [x1, y1, x2, y2]
        self.feature = None  # appearance feature vector
        self.missed = 0  # number of consecutive misses
        self.hits = 1  # total number of hits
        
        
        
    
    def _init_kf(self, dt: float):
        # State transition matrix (constant velocity)
        F = np.eye(8)
        for i in range(4):
            F[i, i+4] = dt
        self.kf.F = F
        
        # Measurement matrix: observe [cx, cy, w, h]
        self.kf.H = np.zeros((4, 8))
        self.kf.H[:, :4] = np.eye(4)

        # 1) Initialize covariance P (start uncertain about velocity)
        
        #    FilterPy default P is eye(dim_x) so we just scale it up
        self.kf.P *= 1000.0

        # 2) Measurement noise R

        #    Higher value → trust measurements less
        self.kf.R = np.eye(4) * 10.0

        # 3) Process noise Q
        #    Small value → assume near‐constant velocity
        self.kf.Q = np.eye(self.kf.dim_x) * 0.01
    

    def predict(self) -> List[int]:
        #1. predict
        self.kf.predict()

        # Convert state → bbox
        self.bbox = state2bbox(self.kf.x.flatten())
        
        self.missed +=1
        return self.bbox


    def update(self, detection_bbox: List[int], feature: np.ndarray) -> List[int]:
        #1. bbbox into a 4x1 vect
        z = bbox2state(detection_bbox).reshape(4,1)

        #2. update the kf
        self.kf.update(z)

        #convert to bbox after the corrected stat
        self.bbox = state2bbox(self.kf.x.flatten())

        self.feature = feature
        self.missed = 0
        #update the amount of tiems a track has been matched to a detection
        self.hits += 1
        
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

        # 2) Build cost matrix & gatej
        cost = build_cost_matrix(self.tracks, detections, self.w_motion, self.w_app)
        index = cost>self.gate_thresh
        cost[index] = 1e6

        # 3) Associate
        matches, unmatched_tracks, unmatched_dets = assign_detections(cost)

        # 4) Update matched
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx], features[det_idx])

        # 5) Handle unmatched tracks
        self.tracks = [
            trk for trk in self.tracks
            if trk.missed <= self.max_age
        ]
        # 6) Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_trk = Track(self.next_id, detections[det_idx])
            new_trk.feature = features[det_idx]
            self.tracks.append(new_trk)
            self.next_id += 1

        # 7. Filter by minimum age/hits
        output_tracks = [
            trk for trk in self.tracks
            if trk.hits >= self.min_hits
        ]

        return output_tracks

#ROS NODE
class TrackingNode(Node):
    def __init__(self):
            super().__init__('tracking_node')
            self.bridge = CvBridge()
            self.yolo = YOLO('yolov8n.pt')
            #initiate track manager
            self.track_manager = TrackManager(
                max_age = 30, #Deletes the track if it hasnt been updated in 30 frames
                min_hits = 3, #Only publishes tracks that have been updated at least 3
                gate_thresh = 9.4877, 
                w_motion = 0.5, #Weight for motion cost
                w_app = 0.5 #Weight for appearance cost
            )
            self.sub_det = self.create_subscription(DetectionArray, '/kestrel/detections', self.dets_cb, 10)
            self.pub_tracks = self.create_publisher(TrackArray, '/kestrel/tracks', 10)
            # Initialize Kalman filter class here
            self.kf = FP_KalmanFilter(dim_x=8, dim_z=4)

    def dets_cb(self, det_msg:DetectionArray):
        # 1. Predict all tracks
        self.track_manager.predict_all()
        # extract appearance features from the cnn
        detection_bboxes = []
        detection_features = []
        for det in det_msg.detections:
            #Extract bounding box
            bbox = det2bbox(det)
            detection_bboxes.append(bbox)

            #Extract appearance feature (dummy example, replace with actual feature extraction)
            #Using placeholders for now, will replace with actual CNN later
            feature = np.random.rand(128)
            detection_features.append(feature)
            
        # 2. Gate & associate (week-2) (Extract bboxes)
        # Cost matrix comparing all tracks to all detections
        cost_matrix = build_cost_matrix(self.track_manager.tracks, detection_bboxes, self.track_manager.w_motion, self.track_manager.w_app)
        # Apply gating - high cost for invalid assignments
        if len(self.track_manager.tracks) > 0 and len(detection_bboxes) > 0:
            for i, track in enumerate(self.track_manager.tracks):
                for j, det_bbox in enumerate(detection_bboxes):
                    # Calculate innovation covariance for gating
                    innovation_cov = track.kf.H @ track.kf.P @ track.kf.H.T + track.kf.R
                    
                    # Check if assignment is valid using Mahalanobis distance
                    if not mahalanobis_gate(track.kf, det_bbox, innovation_cov, self.track_manager.gate_thresh):
                        cost_matrix[i, j] = 1e6
        
        #Hungarian algorithm to find optimal assignments
        matches, unmatched_tracks, unmatched_dets = assign_detections(cost_matrix)
        

        # 3. Update tracks or create new tracks
        
        #we assume that each track has the following features that we must update or delete based on whether the tracker
        #gets matched for N frames

        #Update matched tracks
        for trk_idx, det_idx in matches:
            trk = self.track_manager.tracks[trk_idx]
            #update every feature of a track, meaning the bboxes & features
            trk.update(detection_bboxes[det_idx], detection_features[det_idx])
            
        #Create new tracks for any unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(self.track_manager.next_id, detection_bboxes[det_idx])
            new_track.feature = detection_features[det_idx]
            self.track_manager.tracks.append(new_track)
            self.track_manager.next_id +=1


        #Remove any track with missed > max_age
        self.track_manager.tracks = [
            trk for trk in self.track_manager.tracks
            if trk.missed <= self.track_manager.max_age
        ]


        #4 Build and Publish
        track_array_msg = TrackArray()
        track_array_msg.header = det_msg.header            
        # track_array_msg.tracks.append(to_track_msg(track))  #TODO adding tracks as needed
        for trk in self.track_manager.tracks:
            if trk.hits >= self.track_manager.min_hits:
                track_msg = TrackMsg()
                track_msg.id = trk.id
                track_msg.x1 = float(trk.bbox[0])
                track_msg.y1 = float(trk.bbox[1])
                track_msg.x2 = float(trk.bbox[2])
                track_msg.y2 = float(trk.bbox[3])
                track_array_msg.tracks.append(track_msg)

        self.pub_tracks.publish(track_array_msg)         # publish the instance


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

