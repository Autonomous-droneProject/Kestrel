#This skeleton code's structure comes from the research paper: https://www.mdpi.com/2076-3417/12/3/1319

import numpy as np
from numpy.typing import NDArray

class DataAssociation:
    """
    All bbox arrays are expected in format:
        [cx, cy, w, h]  (center coordinates + width/height)

    All cost matrices returned are shape:
        (T, D) = (num_tracks, num_detections)

    Convention:
        - Lower cost is better
        - Costs are clipped/bounded into [0, 1] for the included costs
    """


    @staticmethod
    def _cxcywh_to_xyxy(boxes: NDArray[np.float_]) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        """
        Convert boxes (N,4) in cxcywh into x1,y1,x2,y2 (each (N,1) or (1,N) depending on reshape).
        """
        boxes = np.asarray(boxes, dtype=np.float32)
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w  = boxes[:, 2]
        h  = boxes[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return x1, y1, x2, y2
    
    #Euclidean Distance Based Cost Matrix (𝐷𝐸(𝐷,𝑃))
    def euclidean_cost(
        self,
        tracks: NDArray[np.float_],        # (T,4) cxcywh
        detections: NDArray[np.float_],    # (D,4) cxcywh
        image_dims: tuple[int, int],
    ) -> NDArray[np.float_]:
        """
        Euclidean distance between centers normalized by half the image diagonal.
        Returns (T, D) in [0, 1] where 0 is best (same center), 1 is worst (far).

        d(Di, Pi) = 1 - [sqrt((u_Di - u_Pi)^2 + (v_Di - v_Pi)^2) / (1/2) * sqrt(h^2 + w^2)]

        where (h, w) are the height and width of the input image.
        """
        tracks = np.asarray(tracks, dtype=np.float32)
        detections = np.asarray(detections, dtype=np.float32)

        if tracks.size == 0 or detections.size == 0:
            return np.zeros((tracks.shape[0], detections.shape[0]), dtype=np.float32)

        # Centers are already (cx, cy)
        trk_pos = tracks[:, 0:2]        # (T,2)
        det_pos = detections[:, 0:2]    # (D,2)

        # Normalization constant: half the image diagonal
        H, W = image_dims
        norm = 0.5 * np.sqrt(float(H) ** 2 + float(W) ** 2)

        # Broadcast subtract:
        #   trk_pos[:, None, :] -> (T,1,2)
        #   det_pos[None, :, :] -> (1,D,2)
        delta = trk_pos[:, None, :] - det_pos[None, :, :]   # (T,D,2)
        dist = np.linalg.norm(delta, axis=2)                # (T,D)

        cost = np.clip(dist / (norm + 1e-12), 0.0, 1.0).astype(np.float32)  # (T,D)
        return cost

    #Bounding Box Ratio Based Cost Matrix (𝑅(𝐷,𝑃))
    def bbox_ratio_cost(
        self,
        tracks: NDArray[np.float_],        # (T,4) cxcywh
        detections: NDArray[np.float_],    # (D,4) cxcywh
    ) -> NDArray[np.float_]:

        """
        Computes the bounding box ratio-based cost matrix (𝑅(𝐷,𝑃)), which is
        implemented as a ratio between the product of each width and height.

        r(Di, Pi) = min( (w_Di * h_Di) / (w_Pi * h_Pi), (w_Pi * h_Pi) / (w_Di * h_Di) )

        cost = 1 - min(area_det/area_trk, area_trk/area_det)

        Note: this is area-only, so shapes like 10x1 and 1x10 have same area.
        """ 
        tracks = np.asarray(tracks, dtype=np.float32)
        detections = np.asarray(detections, dtype=np.float32)

        T = tracks.shape[0]
        D = detections.shape[0]
        if T == 0 or D == 0:
            return np.zeros((T, D), dtype=np.float32)

        trk_areas = tracks[:, 2] * tracks[:, 3]        # (T,)
        det_areas = detections[:, 2] * detections[:, 3]  # (D,)

        eps = 1e-6
        trk_areas = np.maximum(trk_areas, eps)
        det_areas = np.maximum(det_areas, eps)

        # Broadcast to (T,D)
        # trk_areas[:,None] => (T,1)
        # det_areas[None,:] => (1,D)
        ratio1 = det_areas[None, :] / trk_areas[:, None]   # (T,D)  det/trk
        ratio2 = trk_areas[:, None] / det_areas[None, :]   # (T,D)  trk/det

        cost = 1.0 - np.minimum(ratio1, ratio2)
        cost = np.clip(cost, 0.0, 1.0).astype(np.float32)
        return cost

      
    # SORT's IoU Cost
    def iou_cost(
        self,
        tracks: NDArray[np.float_],        # (T,4) cxcywh
        detections: NDArray[np.float_],    # (D,4) cxcywh
    ) -> NDArray[np.float_]:
        """
        IoU cost = 1 - IoU, returned as (T, D).
        """
        tracks = np.asarray(tracks, dtype=np.float32)
        detections = np.asarray(detections, dtype=np.float32)

        T = tracks.shape[0]
        D = detections.shape[0]
        if T == 0 or D == 0:
            return np.zeros((T, D), dtype=np.float32)

        # Convert both to xyxy
        trk_x1, trk_y1, trk_x2, trk_y2 = self._cxcywh_to_xyxy(tracks)      # each (T,)
        det_x1, det_y1, det_x2, det_y2 = self._cxcywh_to_xyxy(detections)  # each (D,)

        # Reshape for broadcasting:
        # Tracks along rows (T,1), detections along cols (1,D)
        trk_x1 = trk_x1[:, None]
        trk_y1 = trk_y1[:, None]
        trk_x2 = trk_x2[:, None]
        trk_y2 = trk_y2[:, None]

        det_x1 = det_x1[None, :]
        det_y1 = det_y1[None, :]
        det_x2 = det_x2[None, :]
        det_y2 = det_y2[None, :]

        inter_x1 = np.maximum(trk_x1, det_x1)
        inter_y1 = np.maximum(trk_y1, det_y1)
        inter_x2 = np.minimum(trk_x2, det_x2)
        inter_y2 = np.minimum(trk_y2, det_y2)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h  # (T,D)

        trk_area = np.maximum(0.0, (trk_x2 - trk_x1)) * np.maximum(0.0, (trk_y2 - trk_y1))  # (T,1)
        det_area = np.maximum(0.0, (det_x2 - det_x1)) * np.maximum(0.0, (det_y2 - det_y1))  # (1,D)

        union = trk_area + det_area - inter_area  # (T,D)

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0)

        iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
        iou = np.clip(iou, 0.0, 1.0).astype(np.float32)

        cost = (1.0 - iou).astype(np.float32)  # (T,D)
        return cost

    #SORT's IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix (𝐸𝐼𝑜𝑈𝐷(𝐷,𝑃))
    def iou_euclidean_cost(self, detections, tracks, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix using the Hadamard (element-wise) product:

        EIoUD(D, P) = IoU(D, P) ∘ DE(D, P)

        where ∘ represents element-wise multiplication.
        """
        #Call iou cost matrix
        iou_matrix = np.array(self.iou_cost(detections, tracks))

        #Call euclidean cost matrix
        euclidean_matrix = np.array(self.euclidean_cost(detections, tracks, image_dims))

        #Perform Hadamard product
        iou_euclidean_cost_matrix = iou_matrix * euclidean_matrix

        return iou_euclidean_cost_matrix


    #SORT’s IoU Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐼𝑜𝑈(𝐷,𝑃))
    def iou_bbox_ratio_cost(self, detections, tracks):
        """
        Computes the IoU cost matrix combined with the bounding box ratio-based
        cost matrix using the Hadamard (element-wise) product:

        RIoU(D, P) = IoU(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        num_detections, num_tracks = len(detections), len(tracks)

        if num_detections == 0 or num_tracks == 0:
            return np.array([])
        
        cost_iou = np.asarray(self.iou_cost(detections, tracks))  # IoU cost matrix
        cost_bbr = np.asarray(self.bbox_ratio_cost(detections, tracks))  # Bounding box ratio cost matrix

        if np.shape(cost_iou) != np.shape(cost_bbr):
            raise ValueError("IoU cost matrix and bbox ratio cost matrix are of different shapes")

        return cost_iou * cost_bbr  # Element-wise multiplication
   

    #Euclidean Distance Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐷𝐸(𝐷,𝑃))
    def euclidean_bbox_ratio_cost(self, detections, tracks, image_dims):
        """
        Computes the Euclidean distance cost matrix combined with the bounding box
        ratio-based cost matrix using the Hadamard (element-wise) product:
        RDE(D, P) = DE(D, P) ∘ R(D, P)
        where ∘ represents element-wise multiplication.
        """
        num_detections, num_tracks = len(detections), len(tracks)
        
        if num_detections == 0 or num_tracks == 0:
            return np.array([])
        
        cost_de = np.asarray(self.euclidean_cost(detections, tracks, image_dims))
        cost_r = np.asarray(self.bbox_ratio_cost(detections, tracks))
        
        if np.shape(cost_de) != np.shape(cost_r):
            raise ValueError("Euclidean cost matrix and bbox ratio cost matrix are of different shapes")
        
        # performs element-wise multiplication
        cost_rde = np.multiply(cost_de, cost_r)

        return cost_rde

      
    #Step 7: SORT's IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix and the Bounding Box Ratio Based Cost Matrix (𝑀(𝐷,𝑃))
    def combined_cost_matrix(self, detections, tracks, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix and the bounding box ratio-based cost matrix using the Hadamard
        (element-wise) product:

        M(D, P) = IoU(D, P) ∘ DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """

        num_detections = len(detections)
        num_tracks = len(tracks)

        if num_detections == 0 or num_tracks == 0:
            return np.array([])
        
        matrix_iou = self.iou_cost(detections, tracks)
        matrix_de = self.euclidean_cost(detections, tracks, image_dims)
        matrix_r = self.bbox_ratio_cost(detections, tracks)

        # Ensure all matrices have the same shape
        if not (matrix_iou.shape == matrix_de.shape == matrix_r.shape == (num_detections, num_tracks)):
            raise ValueError("Cost matrices must have the same shape.")
        
        # Compute the combined cost matrix using element-wise multiplication
        # Each component is already a cost (lower = better), so multiplication is safe
        combined_matrix = matrix_iou * matrix_de * matrix_r

        return combined_matrix
    
        
    #Element-wise Average of Every Cost Matrix (𝐴(𝐷,𝑃))
    def average_cost_matrix(self, detections, tracks, image_dims):
        """
        Computes the element-wise average of every cost matrix:

        A(Di, Pi) = (IoU(Di, Pj) + DE(Di, Pj) + R(Di, Pj)) / 3,  for i ∈ D, j ∈ P
        """
        num_detections = len(detections)
        num_tracks = len(tracks)

        if num_detections == 0 or num_tracks == 0:
            return np.array([]) #Return an empty array if there are no tracks or detections

        cost_iou = self.iou_cost(detections, tracks)
        cost_de = self.euclidean_cost(detections, tracks, image_dims)
        cost_r = self.bbox_ratio_cost(detections, tracks)

        # Ensure all shapes match
        if not (cost_iou.shape == cost_de.shape == cost_r.shape):
            raise ValueError("Cost matrices are not aligned in shape.")
    
        #Final cost matrix
        return (cost_iou + cost_de + cost_r) / 3
    

    #Element-wise Weighted Mean of Every Cost Matrix Value (𝑊𝑀(𝐷,𝑃))
    def weighted_mean_cost_matrix(
        self,
        tracks: NDArray[np.float_],        # (T,4) cxcywh
        detections: NDArray[np.float_],    # (D,4) cxcywh
        image_dims: tuple[int, int],
        lambda_iou: float = 7 / 10,
        lambda_de: float = 2 / 10,
        lambda_r: float = 1 / 10,
    ) -> NDArray[np.float_]:
        """
        WM = lambda_iou * IoU_cost + lambda_de * Euclidean_cost + lambda_r * Ratio_cost
        Returns (T, D).
        """
        tracks = np.asarray(tracks, dtype=np.float32)
        detections = np.asarray(detections, dtype=np.float32)

        T = tracks.shape[0]
        D = detections.shape[0]
        if T == 0 or D == 0:
            return np.zeros((T, D), dtype=np.float32)

        # Ensure weights sum to 1.0
        s = float(lambda_iou + lambda_de + lambda_r)
        if not np.isclose(s, 1.0):
            lambda_iou /= s
            lambda_de  /= s
            lambda_r   /= s

        cost_iou = self.iou_cost(tracks, detections)                    # (T,D)
        cost_de  = self.euclidean_cost(tracks, detections, image_dims)  # (T,D)
        cost_r   = self.bbox_ratio_cost(tracks, detections)             # (T,D)

        wm = (lambda_iou * cost_iou) + (lambda_de * cost_de) + (lambda_r * cost_r)
        wm = np.clip(wm, 0.0, 1.0).astype(np.float32)
        return wm

      
    #Class Gate Update Based on Object Class Match (𝐶∗(𝐷,𝑃))
    def class_gate_cost_matrix(self, cost_matrix, detection_classes, track_classes):
        """
        Updates the cost matrix based on the match between predicted and detected
        object class. If the class labels do not match, the cost is set to 0:

        C*(Ci, j, Di, Pi) = { Ci, j if Class_Di = Class_Pi, 0 otherwise }

        for i ∈ D, j ∈ P.
        """
        '''
        This function updates cost matrices based on the match between 
        predicted and detected object classes'''
        
        num_detections = cost_matrix.shape[0]
        num_tracks = cost_matrix.shape[1]

        if num_detections != len(detection_classes) or num_tracks != len(track_classes):
            raise ValueError("Dimensions of cost_matrix, detection_classes, and track_classes do not match - Class Gate Cost Matrix")

        #Create a boolean mask where classses match
        #            Reshapes to (num_detections, 1)      Reshapes to (1, num_tracks) -> (D, T)
        match_mask = (np.array(detection_classes)[:, None] == np.array(track_classes)[None, :]) #Shape = [num_detections, num_tracks]
        #Because detection_classes has the same number of rows as track_classes has columns, we can perform matrix multiplication
        
        #Apply the mask and keep the values where classes match, zero where they do not
        gated_cost_matrix = cost_matrix * match_mask
        
        return gated_cost_matrix
    

_DA = DataAssociation()

def weighted_mean_cost_matrix(
    tracks: NDArray[np.float_],
    detections: NDArray[np.float_],
    image_dims: tuple[int, int],
    lambda_iou: float = 7 / 10,
    lambda_de: float = 2 / 10,
    lambda_r: float = 1 / 10,
) -> NDArray[np.float_]:
    """
    Pure NumPy in/out helper.
    Expects:
      tracks: (T,4) cxcywh
      detections: (D,4) cxcywh
    Returns:
      (T,D) cost matrix
    """
    return _DA.weighted_mean_cost_matrix(
        tracks=tracks,
        detections=detections,
        image_dims=image_dims,
        lambda_iou=lambda_iou,
        lambda_de=lambda_de,
        lambda_r=lambda_r,
    )