import rclpy
from rclpy.node import Node

from kestrel_msgs.msg import DetectionArray, Track as TrackMsg, TrackArray
from kestrel_tracking.DataAssociation import weighted_mean_cost_matrix

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
    conf: float = 0.0
    class_name: str = ""

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
        conf=float(det_msg.conf), 
        class_name=str(det_msg.class_name),
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
# Costs + gating (vectorized)
# -----------------------------
def mahalanobis_gate_mask(
    tracks: List["Track"],
    det_z: np.ndarray,
    gate_d2_thresh: float
) -> np.ndarray:
    """
    Build a (T,D) boolean mask where True means the pairing is allowed by gating.

    Implementation detail:
      - We loop over tracks (T loops)
      - We vectorize over detections per-track (no T*D Python loops)

    Args:
      tracks: list of Track objects
      det_z: (D,4) array of detection measurements in cxcywh
      gate_d2_thresh: squared Mahalanobis threshold

    Returns:
      mask: (T,D) boolean array
    """
    T = len(tracks)
    D = det_z.shape[0]
    mask = np.zeros((T, D), dtype=bool)

    if T == 0 or D == 0:
        return mask

    # Ensure det_z is float32 for speed/consistency
    det_z = np.asarray(det_z, dtype=np.float32)

    # Precompute det_z as (4,D) once, so each track can reuse it cheaply
    # det_z_T has columns = detections
    det_z_T = det_z.T  # (4,D)

    for i, trk in enumerate(tracks):
        kf = trk.kf

        # Innovation covariance S = HPH^T + R (4x4)
        H = kf.H.astype(np.float32)
        P = kf.P.astype(np.float32)
        R = kf.R.astype(np.float32)
        S = H @ P @ H.T + R  # (4,4)

        # Predicted measurement Hx (4,1)
        Hx = (H @ kf.x).astype(np.float32)  # (4,1)

        # Innovations for all detections at once:
        # y = z - Hx, but z is (4,D) and Hx is (4,1) so broadcast works
        y = det_z_T - Hx  # (4,D)

        # Solve S^{-1} y for all columns at once (4,D)
        # This avoids explicitly computing inv(S)
        try:
            tmp = np.linalg.solve(S, y)  # (4,D)
        except np.linalg.LinAlgError:
            # If S becomes singular for any reason, fail-safe: gate nothing for this track
            continue

        # d2 per detection = y^T * S^{-1} * y
        # Since y and tmp are (4,D), columnwise dot is sum(y * tmp) over rows
        d2 = np.sum(y * tmp, axis=0)  # (D,)

        mask[i, :] = (d2 <= gate_d2_thresh)

    return mask


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

def appearance_cost_matrix(
    track_feats: np.ndarray,
    det_feats: np.ndarray,
    track_has_feat: np.ndarray
) -> np.ndarray:
    """
    Cosine distance matrix (T,D):
      cost = 1 - cosine_similarity

    If a track has no feature, we force its cost row to 1.0 (max cost),
    matching your old scalar behavior.

    Args:
      track_feats: (T,F) float32
      det_feats: (D,F) float32
      track_has_feat: (T,) bool

    Returns:
      (T,D) float32 matrix in [0,2] but practically [0,1] if normalized; we clip to [0,1].
    """
    T = track_feats.shape[0]
    D = det_feats.shape[0]

    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    # Normalize features (avoid divide by 0)
    eps = 1e-12

    trk = track_feats.astype(np.float32, copy=False)
    det = det_feats.astype(np.float32, copy=False)

    trk_norm = trk / (np.linalg.norm(trk, axis=1, keepdims=True) + eps)  # (T,F)
    det_norm = det / (np.linalg.norm(det, axis=1, keepdims=True) + eps)  # (D,F)

    # Cosine similarity: (T,D) = (T,F) @ (F,D)
    sim = trk_norm @ det_norm.T  # (T,D)

    # Convert to cosine distance cost
    cost = (1.0 - sim).astype(np.float32)

    # Tracks without features => max cost
    # We set the entire row to 1.0 (same as your scalar function returning 1.0)
    if track_has_feat is not None and track_has_feat.size == T:
        cost[~track_has_feat, :] = 1.0

    # Keep bounded for sanity
    cost = np.clip(cost, 0.0, 1.0)
    return cost


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
        self.conf = init_det.conf            # Store conf
        self.class_name = init_det.class_name # Store class
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
        self.conf = det.conf                 # Update conf
        self.class_name = det.class_name    # Not sure how this could happen but...
        self.feature = det.embedding
        self.missed = 0
        # Update the amount of times a track has been matched to a detection
        self.hits += 1

    def z_pred(self) -> np.ndarray:
        """
        Predicted measurement (4,) in cxcywh:
          z_pred = Hx
        Using this keeps motion costs consistent with Mahalanobis gating space.
        """
        z = (self.kf.H @ self.kf.x).reshape(4,)
        return z.astype(np.float32, copy=False)
    

class TrackManager:
    def __init__(
        self, 
        max_age: int, 
        min_hits: int, 
        gate_d2_thresh: float, 
        w_motion: float, 
        w_app: float,
        dt: float = 1.0,
        image_dims: tuple[int, int] = (720, 1280),  # <--- TODO: placeholder dims, must verify
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

        self.image_dims = image_dims


        # ---- tiny adapters (for use with DataAssociation) ----
    @staticmethod
    def _stack_track_z(tracks: List[Track]) -> np.ndarray:
        """
        Stack predicted track measurements into (T,4) cxcywh.
        """
        if len(tracks) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([t.z_pred() for t in tracks], axis=0).astype(np.float32)

    @staticmethod
    def _stack_det_z(detections: List[Det]) -> np.ndarray:
        """
        Stack detection measurements into (D,4) cxcywh.
        """
        if len(detections) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([d.z.reshape(4,) for d in detections], axis=0).astype(np.float32)

    @staticmethod
    def _stack_track_features(tracks: List[Track], feat_dim: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """
        Stack track features into (T,F). If a track has no feature, fill zeros and mark it invalid.

        Returns:
          feats: (T,F) float32
          has_feat: (T,) bool
        """
        T = len(tracks)
        if T == 0:
            return np.zeros((0, feat_dim), dtype=np.float32), np.zeros((0,), dtype=bool)

        feats = np.zeros((T, feat_dim), dtype=np.float32)
        has_feat = np.zeros((T,), dtype=bool)

        for i, t in enumerate(tracks):
            if t.feature is None:
                continue
            feats[i, :] = t.feature.astype(np.float32, copy=False)
            has_feat[i] = True

        return feats, has_feat

    @staticmethod
    def _stack_det_features(detections: List[Det], feat_dim: int = 128) -> np.ndarray:
        """
        Stack detection embeddings into (D,F).
        """
        D = len(detections)
        if D == 0:
            return np.zeros((0, feat_dim), dtype=np.float32)
        return np.stack([d.embedding.astype(np.float32, copy=False) for d in detections], axis=0)
    

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
        
        # ---- adapters (build arrays once per frame) ----
        Zt = self._stack_track_z(self.tracks)      # (T,4) cxcywh
        Zd = self._stack_det_z(detections)         # (D,4) cxcywh

        Ft, track_has_feat = self._stack_track_features(self.tracks, feat_dim=128)  # (T,128), (T,)
        Fd = self._stack_det_features(detections, feat_dim=128)                      # (D,128)

        # ---- vectorized motion cost via DataAssociation (returns (T,D)) ----
        mc = weighted_mean_cost_matrix(
            tracks=Zt,
            detections=Zd,
            image_dims=self.image_dims,
            # You can pass custom lambdas here if you want:
            # lambda_iou=0.7, lambda_de=0.2, lambda_r=0.1
        ).astype(np.float32, copy=False)  # (T,D)

        # ---- vectorized appearance cost (returns (T,D)) ----
        ac = appearance_cost_matrix(Ft, Fd, track_has_feat).astype(np.float32, copy=False)  # (T,D)

        # ---- gating mask (returns (T,D) bool) ----
        gate = mahalanobis_gate_mask(self.tracks, Zd, self.gate_d2_thresh)  # (T,D)

        # ---- combine costs ----
        cost = (self.w_motion * mc) + (self.w_app * ac)  # (T,D)

        # Any disallowed pairing becomes "impossible"
        cost = cost.astype(np.float32, copy=False)
        cost[~gate] = 1e6

        # 3) Hungarian assignment
        matches, unmatched_tracks, unmatched_dets = hungarian_assign(cost)

        # 4) Update matched tracks
        for trk_idx, det_idx in matches:
            if cost[trk_idx, det_idx] >= 1e6:
                unmatched_tracks.append(trk_idx)
                unmatched_dets.append(det_idx)
                continue
            self.tracks[trk_idx].update(detections[det_idx])

        # 5) Unmatched tracks: do nothing (they already had missed++ in predict())
        unmatched_dets = sorted(set(unmatched_dets))

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

            image_dims = (720, 1280)  # (H,W) <-- TODO: set to your camera feed size, consider as ROS params 

            self.track_manager = TrackManager(
                max_age=30, # Deletes the track if it hasnt been updated in 30 frames
                min_hits=1, # Only publishes tracks that have been updated at least 3
                gate_d2_thresh=9.4877,   # chi2.ppf(0.95, df=4) is ~9.49
                w_motion=0.5, #W eight for motion cost
                w_app=0.5, # Weight for appearance cost
                dt=1.0,
                image_dims=image_dims,
            )
            
            self.sub_det = self.create_subscription(
                DetectionArray, 
                '/kestrel/detections', 
                self.dets_cb, 
                10,
            )
            self.pub_tracks = self.create_publisher(TrackArray, '/kestrel/tracks', 10)

    def publish_tracks(self, header, tracks: List[Track]) -> None:
        """Turn internal tracks into TrackArray message and publish."""
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
            
            t.center_x = (t.x1 + t.x2) / 2.0
            t.center_y = (t.y1 + t.y2) / 2.0

            t.conf = float(trk.conf)
            t.class_name = str(trk.class_name)

            # Copy the embedding from the internal Track object to the ROS message
            if trk.feature is not None:
                # trk.feature is a numpy array, convert to list for ROS
                t.embedding = trk.feature.tolist()
                
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