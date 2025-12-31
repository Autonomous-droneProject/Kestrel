import rclpy
from rclpy.node import Node

from kestrel_msgs.msg import DetectionArray, Track as TrackMsg, TrackArray

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from scipy.optimize import linear_sum_assignment

from filterpy.kalman import KalmanFilter as FP_KalmanFilter



# -----------------------------
# Detection struct + adapters
# -----------------------------
@dataclass(frozen=True)
class Det:
    """One detection in convenient tracker format."""
    x1: float
    y1: float
    x2: float
    y2: float
    z: np.ndarray          # (4,1) measurement: [cx, cy, w, h]^T
    embedding: np.ndarray  # (128,) appearance embedding

def det_from_ros(det_msg) -> Det:
    """Convert kestrel_msgs/Detection -> Det."""
    z = np.array(
        [det_msg.center_x, det_msg.center_y, det_msg.w, det_msg.h], 
        dtype=np.float32
    ).reshape(4,1)
    
    emb = np.array(det_msg.embedding, dtype=np.float32)
    
    return Det(
        x1=float(det_msg.x1), 
        y1=float(det_msg.y1), 
        x2=float(det_msg.x2), 
        y2=float(det_msg.y2), 
        z=z, 
        embedding=emb,
    )


# -----------------------------
# Geometry helpers
# -----------------------------
def state2bbox_xyxy(state: Sequence[float]) -> List[float]:
    """
    Convert a KF state -> bbox [x1, y1, x2, y2] using state[:4] = [cx, cy, w, h].
    Keep as floats; cast to int only for drawing.
    """
    cx, cy, w, h = state[:4]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [float(x1), float(y1), float(x2), float(y2)]


# -----------------------------
# Costs + gating
# -----------------------------
def mahalanobis_distance_squared(kf: FP_KalmanFilter, z: np.ndarray) -> float:
    """
    Compute squared Mahalanobis distance:
      d^2 = (z - Hx)^T S^{-1} (z - Hx),
    where S = HPH^T + R.
    """
    H = kf.H
    x = kf.x
    y = z - (H @ x)             # innovation
    S = H @ kf.P @ H.T + kf.R   # innovation covariance
    
    # Solve S^{-1} y without explicitly inverting S
    tmp = np.linalg.solve(S, y)
    d2 = float((y.T @ tmp).item())
    return d2


def appearance_cost(track_feature: np.ndarray, det_feature: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity."""
    if track_feature is None:
        return 1.0  # max cost if no feature
    f1 = track_feature / (np.linalg.norm(track_feature) + 1e-12)
    f2 = det_feature / (np.linalg.norm(det_feature) + 1e-12)
    return float(1.0 - np.dot(f1, f2))

#TODO: someone forgot this
def motion_cost(track: "Track", det: Det) -> float:
    """
    Placeholder for now.
    Later this becomes IoU/ WM / etc. from DataAssociation.py
    Must return a cost where lower is better.
    """
    return 0.0



# -----------------------------
# Assignment
# -----------------------------
def hungarian_assign(cost: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve Hungarian for minimal assignment.
    Returns:
        matches: (track_idx, det_idx)
        unmatched_tracks: [track_idx]
        unmatched_dets: [det_idx]
    """
    if cost.size == 0:
        # Nothing to match
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    

    # Run Hungarian
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build the matched pairs
    matches = list(zip(row_ind.tolist(), col_ind.tolist()))

    # Determine unmatched pairs 
    all_tracks = set(range(cost.shape[0]))
    all_dets   = set(range(cost.shape[1]))
    matched_tracks = set(row_ind.tolist())
    matched_dets   = set(col_ind.tolist())

    unmatched_tracks = sorted(list(all_tracks - matched_tracks))
    unmatched_dets   = sorted(list(all_dets - matched_dets))

    return matches, unmatched_tracks, unmatched_dets


# -----------------------------
# Track + TrackManager
# -----------------------------
class Track:
    """
    KF state is [cx, cy, w, h, vx, vy, vw, vh]^T
    Measurement z is [cx, cy, w, h]^T
    """
    def __init__(self, track_id: int, init_det: Det, dt: float = 1.0):
        self.id = int(track_id)
        self.kf = FP_KalmanFilter(dim_x=8, dim_z=4)
        self._init_kf(dt)

        # Initialize state from detection measurement
        cx, cy, w, h = init_det.z.flatten().tolist()
        self.kf.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(8, 1)

        #Initialize other attributes
        self.bbox_xyxy: List[float] = [init_det.x1, init_det.y1, init_det.x2, init_det.y2]
        self.feature: np.ndarray | None = init_det.embedding

        self.missed: int = 0  # number of consecutive misses
        self.hits: int = 1  # total number of hits
        

    def _init_kf(self, dt: float) -> None:
        # State transition matrix (constant velocity)
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt
        self.kf.F = F
        
        # Measurement matrix: observe [cx, cy, w, h]
        H = np.zeros((4, 8), dtype=np.float32)
        H[:, :4] = np.eye(4, dtype=np.float32)
        self.kf.H = H

        #TODO: Tune Values Later
        # Initialize covariance P (start uncertain about velocity)
        self.kf.P *= 1000.0
        
        # Measurement noise R
        # Higher value -> trust measurements less
        self.kf.R = np.eye(4, dtype=np.float32) * 10.0

        # Process noise Q
        # Small value -> assume near‐constant velocity
        self.kf.Q = np.eye(8, dtype=np.float32) * 0.01
    

    def predict(self) -> None:
        self.kf.predict()
        self.bbox_xyxy = state2bbox_xyxy(self.kf.x.flatten())
        self.missed +=1     # increment; reset to 0 on successful update


    def update(self, det: Det) -> None:
        self.kf.update(det.z)
        self.bbox_xyxy = state2bbox_xyxy(self.kf.x.flatten())

        self.feature = det.embedding
        self.missed = 0
        # Update the amount of times a track has been matched to a detection
        self.hits += 1


class TrackManager:
    def __init__(
        self, 
        max_age: int, 
        min_hits: int, 
        gate_d2_thresh: float, 
        w_motion: float, 
        w_app: float,
        dt: float = 1.0,
    ):
        self.tracks: List[Track] = []
        self.next_id: int = 0

        self.max_age = int(max_age)
        self.min_hits = int(min_hits)

        # This threshold is squared Mahalanobis distance
        self.gate_d2_thresh = float(gate_d2_thresh)

        self.w_motion = float(w_motion)
        self.w_app = float(w_app)
        self.dt = float(dt)


    def step(self, detections: List[Det]) -> List[Track]:
        # 1) Predict existing tracks forward
        for trk in self.tracks:
            trk.predict()

        # If no tracks exist, create from all detections
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det, dt=self.dt))
                self.next_id += 1
            return [t for t in self.tracks if t.hits >= self.min_hits]
        
        # If no detections, just age/prune tracks
        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]
        

        # 2) Build cost matrix (tracks x detections) with Mahalanobis gating
        N = len(self.tracks)
        M = len(detections)
        cost = np.full((N, M), 1e6, dtype=np.float32) # default = impossible

        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # Gate using squared Mahalanobis distance in measurement space
                d2 = mahalanobis_distance_squared(trk.kf, det.z)
                if d2 > self.gate_d2_thresh:
                    continue    # leave as 1e6 (invalid assignment)

                mc = motion_cost(trk, det)  # placeholder for now
                ac = appearance_cost(trk.feature, det.embedding)
                cost[i, j] = self.w_motion * mc + self.w_app * ac
            

        # 3) Hungarian assignment
        matches, unmatched_tracks, unmatched_dets = hungarian_assign(cost)

        # 4) Update matched tracks
        for trk_idx, det_idx in matches:
            if cost[trk_idx, det_idx] >= 1e6:
                # This pairing was effectively invalid; treat as unmatched instead
                unmatched_tracks.append(trk_idx)
                unmatched_dets.append(det_idx)
                continue
            self.tracks[trk_idx].update(detections[det_idx])

        # 5) Unmatched tracks: do nothing (they already had missed++ in predict())
        # unmatched_tracks is still useful for debugging/logging, but state aging is handled.
        unmatched_dets = sorted(set(unmatched_dets)) # Safety to de-duplicate before creating new tracks
        
        # 6) Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(Track(self.next_id, detections[det_idx], dt=self.dt))
            self.next_id += 1

        # 7) Prune dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        # 8) Return only publishable tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]



# -----------------------------
# ROS Node
# -----------------------------
class TrackingNode(Node):
    def __init__(self):
            super().__init__('tracking_node')   


            self.track_manager = TrackManager(
                max_age=30, # Deletes the track if it hasnt been updated in 30 frames
                min_hits=3, # Only publishes tracks that have been updated at least 3
                gate_d2_thresh=9.4877,   # chi2.ppf(0.95, df=4) is ~9.49
                w_motion=0.5, #W eight for motion cost
                w_app=0.5, # Weight for appearance cost
                dt=1.0,
            )
            
            self.sub_det = self.create_subscription(
                DetectionArray, 
                '/kestrel/detections', 
                self.dets_cb, 
                10,
            )
            self.pub_tracks = self.create_publisher(TrackArray, '/kestrel/tracks', 10)

    def publish_tracks(self, header, tracks: List[Track]) -> None:
        """Turn internal tracks into TrackArray message and publish"""
        msg = TrackArray()
        msg.header = header

        for trk in tracks:
            t = TrackMsg()
            t.id = int(trk.id)
            # Publish current bbox (xyxy). Keep float.
            t.x1 = float(trk.bbox_xyxy[0])
            t.y1 = float(trk.bbox_xyxy[1])
            t.x2 = float(trk.bbox_xyxy[2])
            t.y2 = float(trk.bbox_xyxy[3])
            msg.tracks.append(t)

        self.pub_tracks.publish(msg)


    def dets_cb(self, det_msg: DetectionArray) -> None:
        dets = [det_from_ros(d) for d in det_msg.detections]
        active_tracks = self.track_manager.step(dets)
        self.publish_tracks(det_msg.header, active_tracks)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()