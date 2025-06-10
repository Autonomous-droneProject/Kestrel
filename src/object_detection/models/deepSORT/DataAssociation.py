#This skeleton code's structure comes from the research paper: https://www.mdpi.com/2076-3417/12/3/1319

#Contain the CNN that deepSORT uses in replace of the Hungarion based Cost Matrix
from filterpy.kalman import KalmanFilter
import numpy as np

class DataAssociation:
    """
    For all parameters:
    tracks : List[List[float]] or List[np.ndarray]
        A list of bounding boxes representing predicted tracks, each in the format [x, y, w, h].
    detections : List[List[float]] or List[np.ndarray]  
         A list of bounding boxes representing current detections, each in the format [x, y, w, h].
    """

    #Euclidean Distance Based Cost Matrix (𝐷𝐸(𝐷,𝑃))
    def euclidean_cost(self, tracks, detections, image_dims):
        """
        Computes the Euclidean distance cost matrix (𝐷𝐸(𝐷,𝑃)), which represents
        the distance between bounding box central points normalized into half
        of the image dimension. To formulate the problem as a maximization
        problem, the distance is obtained by the difference between 1 and the
        normalized Euclidean distance.

        d(Di, Pi) = 1 - sqrt((u_Di - u_Pi)^2 + (v_Di - v_Pi)^2) / (1/2) * sqrt(h^2 + w^2)

        where (h, w) are the height and width of the input image.
        """
        #Retrive lengths
        tracks = np.array(tracks, copy=False)
        detections = np.array(detections, copy=False)
        N_detections = len(detections)
        N_predictions = len(tracks)
        #Store bounding boxes centers for computation
        tracks_pos = tracks[:, 0:2]
        detections_pos = detections[:, 0:2]
        #Calculate norm based off image size
        norm = 0.5 * np.sqrt(image_dims[0]**2 + image_dims[1]**2)
        #Subtract so u_Di - u_Pi & v_Di - v_Pi
        delta = detections_pos[:, None, :] - tracks_pos[None, :, :]
        #Perform linear norm of sum of deltas
        dist_matrix = np.linalg.norm(delta, axis=2)  
        #Compute cost matrix
        euclidean_cost_matrix = 1.0 - (dist_matrix / norm)        
        return euclidean_cost_matrix

    #Bounding Box Ratio Based Cost Matrix (𝑅(𝐷,𝑃))
    def bbox_ratio_cost(tracks, detections):
        """
        Computes the bounding box ratio-based cost matrix (𝑅(𝐷,𝑃)), which is
        implemented as a ratio between the product of each width and height.

        r(Di, Pi) = min( (w_Di * h_Di) / (w_Pi * h_Pi), (w_Pi * h_Pi) / (w_Di * h_Di) )

        Returns a cost matrix where lower values indicate better box shape alignment.

        Box shape similarity ranges from 0 (different) to 1 (identical), and is converted to cost as:
        cost_r = 1.0 - similarity_r.
        
        """ 
        if len(tracks) == 0 or len(detections) == 0:
            return np.array([])

        detections = np.array(detections) # (D, 4)
        tracks = np.array(tracks) # (T, 4)

        # Gets every width and height from each row into 2 1D arrays and calculates area
        detection_areas = detections[:, 2] * detections[:, 3]
        track_areas = tracks[:, 2] * tracks[:, 3] # (T,) = (T,) * (T,)

        # Transform the 1D arrays to broadcast into (D, T)
        detection_areas = detection_areas[:, None]# (D, 1): [[1], [2], [3]] 
        track_areas = track_areas[None, :] #        (1, T): [[1, 2, 3, 4]]

        # Calculates ratio and broadcasts to (D, T)
        ratio1 = detection_areas / track_areas
        ratio2 = track_areas / detection_areas

        # Calculates cost at every [i, j]
        bbox_cost_matrix = 1.0 - np.minimum(ratio1, ratio2)
        return bbox_cost_matrix

      
   #SORT’s IoU Cost Matrix
    def iou_cost(tracks,detections):
        
        numDetections = len(detections)
        numTracks = len(tracks)
        
        
        det_x1 = detections[:, 0:1]
        det_y1 = detections[:, 1:2]
        det_x2 = det_x1 + detections[:, 2:3]
        det_y2 = det_y1 + detections[:, 3:4]

        trk_x1 = tracks[:, 0].reshape(1,-1)
        trk_y1 = tracks[:, 1].reshape(1,-1)
        trk_x2 = trk_x1 + tracks[:, 2].reshape(1,-1)
        trk_y2 = trk_y1 + tracks[:, 3].reshape(1,-1)
            
        detectionWidth = detections[:,2:3]
        detectionHeight = detections[:,3:4]
        
               
        areaDetection = detectionWidth * detectionHeight
        areaTrack = tracks[:, 2].reshape(1, -1) * tracks[:, 3].reshape(1, -1)

        inter_x1 = np.maximum(det_x1, trk_x1)
        inter_y1 = np.maximum(det_y1, trk_y1)
        inter_x2 = np.minimum(det_x2, trk_x2)
        inter_y2 = np.minimum(det_y2, trk_y2)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        
 
        
        "AoI = Area of Intersection, AoU = Area of Union"
        AoI = inter_w * inter_h
        AoU = areaDetection + areaTrack - AoI
       
        iou_matrix = np.where(AoU > 0, AoI / AoU, 0.0)
       
        return 1.0 - iou_matrix

    #   SORT’s IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix (𝐸𝐼𝑜𝑈𝐷(𝐷,𝑃))
    def iou_euclidean_cost(self, tracks, detections, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix using the Hadamard (element-wise) product:

        EIoUD(D, P) = IoU(D, P) ∘ DE(D, P)

        where ∘ represents element-wise multiplication.
        """
        #Call iou cost matrix
        iou_matrix = np.array(self.iou_cost(tracks, detections))
        #Call euclidean cost matrix
        euclidean_matrix = np.array(self.euclidean_cost(tracks, detections, image_dims))
        #Perform Hadamard product
        iou_euclidean_cost_matrix = iou_matrix * euclidean_matrix
        #Return as list
        return iou_euclidean_cost_matrix


    #SORT’s IoU Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐼𝑜𝑈(𝐷,𝑃))
    def iou_bbox_ratio_cost(self, tracks, detections):
        """
        Computes the IoU cost matrix combined with the bounding box ratio-based
        cost matrix using the Hadamard (element-wise) product:

        RIoU(D, P) = IoU(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        num_tracks, num_detections = len(tracks), len(detections)

        if num_tracks == 0 or num_detections == 0:
            return np.array([])
        
        cost_iou = np.asarray(self.iou_cost(detections, tracks))  # IoU cost matrix
        cost_bbr = np.asarray(self.bbox_ratio_cost(tracks, detections))  # Bounding box ratio cost matrix

        if np.shape(cost_iou) != np.shape(cost_bbr):
            raise ValueError("IoU cost matrix and bbox ratio cost matrix are of different shapes")

        return cost_iou * cost_bbr  # Element-wise multiplication
   

    #Euclidean Distance Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐷𝐸(𝐷,𝑃))
    def euclidean_bbox_ratio_cost(tracks, detections, image_dims):
        """
        Computes the Euclidean distance cost matrix combined with the bounding box
        ratio-based cost matrix using the Hadamard (element-wise) product:

        RDE(D, P) = DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

      
    #Step 7: SORT's IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix and the Bounding Box Ratio Based Cost Matrix (𝑀(𝐷,𝑃))
    def combined_cost_matrix(self, tracks, detections, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix and the bounding box ratio-based cost matrix using the Hadamard
        (element-wise) product:

        M(D, P) = IoU(D, P) ∘ DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """

        num_tracks = len(tracks)
        num_detections = len(detections)

        if num_tracks == 0 or num_detections == 0:
            return np.array([])
        
        matrix_iou = self.iou_cost(tracks, detections) #Already lower is better
        matrix_de = self.euclidean_cost(tracks, detections, image_dims)
        matrix_r = self.bbox_ratio_cost(tracks, detections)

        # Ensure all matrices have the same shape
        if matrix_iou.shape != (num_tracks, num_detections) or \
           matrix_de.shape != (num_tracks, num_detections) or \
           matrix_r.shape != (num_tracks, num_detections):
            raise ValueError("Cost matrices must have the same shape.")
        
        # Compute the combined cost matrix using element-wise multiplication
        # Each component is already a cost (lower = better), so multiplication is safe
        combined_matrix = matrix_iou * matrix_de * matrix_r

        return combined_matrix
    
        
    #Element-wise Average of Every Cost Matrix (𝐴(𝐷,𝑃))
    def average_cost_matrix(self, tracks, detections, image_dims):
        """
        Computes the element-wise average of every cost matrix:

        A(Di, Pi) = (IoU(Di, Pj) + DE(Di, Pj) + R(Di, Pj)) / 3,  for i ∈ D, j ∈ P
        """
        num_tracks = len(tracks)
        num_detections = len(detections)

        if num_tracks == 0 or num_detections == 0:
            return np.array([]) #Return an empty array if there are no tracks or detections

        cost_iou = self.iou_cost(tracks,detections)
        euclidean_cost = self.euclidean_cost(tracks, detections, image_dims)

        # Get appearance features
        det_feat = np.array([d.feature for d in detections])
        trk_feat = np.array([t.feature for t in tracks])

        # Normalize vectors
        det_feat = det_feat / np.linalg.norm(det_feat, axis=1, keepdims=True)
        trk_feat = trk_feat / np.linalg.norm(trk_feat, axis=1, keepdims=True)

        # Compute cosine distance matrix (1 - cosine similarity)
        cost_cosine = 1.0 - np.dot(det_feat, trk_feat.T)

        # Ensure all shapes match
        if cost_iou.shape != euclidean_cost.shape or cost_iou.shape != cost_cosine.shape:
            raise ValueError("Cost matrices are not aligned in shape.")
    
        #Final cost matrix
        return (cost_iou + euclidean_cost + cost_cosine) / 3
    
    
    #Element-wise Weighted Mean of Every Cost Matrix Value (𝑊𝑀(𝐷,𝑃))
    def weighted_mean_cost_matrix(self, tracks, detections, image_dims, lambda_iou=0.33, lambda_de=0.33, lambda_r=0.34):
        """
        Computes the element-wise weighted mean of every cost matrix value:

        WM(Di, Pi) = (λ_IoU * IoU(Di, Pi) + λ_DE * DE(Di, Pi) + λ_R * R(Di, Pi))

        where λ_IoU + λ_DE + λ_R = 1.
        """
        '''
        It calculates a combined cost based on the Intersection over Union (IoU), 
        Euclidean distance, and bounding box ratio metrics, using specified weights.
        '''
        num_detections = len(detections)
        num_tracks = len(tracks)
        

        if num_detections == 0 or num_tracks == 0:
            return np.array([]) #Return an empty array if there are no tracks or detections

        cost_matrix = np.zeros((num_detections, num_tracks))

        #Ensure the weights sum to 1.0
        sum_lambdas = lambda_iou + lambda_de + lambda_r
        if not np.isclose(sum_lambdas, 1.0):
            print("Warning: Lambda weights do not sum to 1.0. I will normalize them.")
            lambda_iou /= sum_lambdas
            lambda_de /= sum_lambdas
            lambda_r /= sum_lambdas

        #Compute the cost matrices using other cost functions. All other functions SHOULD return cost matrices, if not change to: 1.0 - output_matrix
        cost_iou = self.iou_cost(tracks,detections) 
        cost_euclidean = self.euclidean_cost(tracks, detections, image_dims)
        cost_ratio = self.bbox_ratio_cost(tracks, detections)

        #Vectorized weight sum. NumPy arrays are implemented in C under the hood.
        #So with these arrays, math operations are executed in compiled C code, not interpreted Python
        #Rather than iterating through with nested loops, we can perform vector/matrix multiplication on the arrays as a whole
        cost_matrix = (
            lambda_iou * cost_iou +
            lambda_de * cost_euclidean +
            lambda_r * cost_ratio
        )
        
        return cost_matrix

      
    #Class Gate Update Based on Object Class Match (𝐶∗(𝐷,𝑃))
    def class_gate_cost_matrix(self, cost_matrix, track_classes, detection_classes):
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

        if num_tracks != (track_classes) or num_detections != len(detection_classes):
            raise ValueError("Dimensions of cost_matrix, track_classes, and detection_classes do not match - Class Gate Cost Matrix")

        #Create a boolean mask where classses match
        #             Reshapes to (num_tracks, 1)     Reshapes to (1, num_detections)
        match_mask = (detection_classes[:, None] == track_classes[None, :]) #Shape = [num_tracks, num_detections]
        #Because detection_classes has the same number of rows as track_classes has columns, we can perform matrix multiplication
        
        #Apply the mask and keep the values where classes match, zero where they do not
        gated_cost_matrix = cost_matrix * match_mask
        
        return gated_cost_matrix