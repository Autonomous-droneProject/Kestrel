#This skeleton code's structure comes from the research paper: https://www.mdpi.com/2076-3417/12/3/1319

#Contain the CNN that deepSORT uses in replace of the Hungarion based Cost Matrix
from filterpy.kalman import KalmanFilter
import numpy as np

class DataAssociation:
    import numpy as np
    from scipy.spatial.distance import cosine

    #Euclidean Distance Based Cost Matrix (𝐷𝐸(𝐷,𝑃))
    def euclidean_cost(tracks, detections, image_dims):
        """
        Computes the Euclidean distance cost matrix (𝐷𝐸(𝐷,𝑃)), which represents
        the distance between bounding box central points normalized into half
        of the image dimension. To formulate the problem as a maximization
        problem, the distance is obtained by the difference between 1 and the
        normalized Euclidean distance.

        d(Di, Pi) = 1 - sqrt((u_Di - u_Pi)^2 + (v_Di - v_Pi)^2) / sqrt(1/2 * (h^2 + w^2))

        where (h, w) are the height and width of the input image.
        """
        pass

    #Bounding Box Ratio Based Cost Matrix (𝑅(𝐷,𝑃))
    def bbox_ratio_cost(tracks, detections):
        """
        Computes the bounding box ratio-based cost matrix (𝑅(𝐷,𝑃)), which is
        implemented as a ratio between the product of each width and height.

        r(Di, Pi) = min( (w_Di * h_Di) / (w_Pi * h_Pi), (w_Pi * h_Pi) / (w_Di * h_Di) )

        This metric gives values closer to 1 for similar box shapes and values
        closer to 0 for significantly different boxes.
        """
        pass

    #SORT’s IoU Cost Matrix
    def iou_cost(tracks, detections):
        """
        Computes the Intersection over Union (IoU) cost matrix between detections
        and predictions. Lower values indicate better matches.
        """
        pass

    #SORT’s IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix (𝐸𝐼𝑜𝑈𝐷(𝐷,𝑃))
    def iou_euclidean_cost(tracks, detections, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix using the Hadamard (element-wise) product:

        EIoUD(D, P) = IoU(D, P) ∘ DE(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #SORT’s IoU Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐼𝑜𝑈(𝐷,𝑃))
    def iou_bbox_ratio_cost(tracks, detections):
        """
        Computes the IoU cost matrix combined with the bounding box ratio-based
        cost matrix using the Hadamard (element-wise) product:

        RIoU(D, P) = IoU(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #Euclidean Distance Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐷𝐸(𝐷,𝑃))
    def euclidean_bbox_ratio_cost(tracks, detections, image_dims):
        """
        Computes the Euclidean distance cost matrix combined with the bounding box
        ratio-based cost matrix using the Hadamard (element-wise) product:

        RDE(D, P) = DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #Step 7: SORT’s IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix and the Bounding Box Ratio Based Cost Matrix (𝑀(𝐷,𝑃))
    def combined_cost_matrix(tracks, detections, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix and the bounding box ratio-based cost matrix using the Hadamard
        (element-wise) product:

        M(D, P) = IoU(D, P) ∘ DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #Element-wise Average of Every Cost Matrix (𝐴(𝐷,𝑃))
    def average_cost_matrix(tracks, detections, image_dims):
        """
        Computes the element-wise average of every cost matrix:

        A(Di, Pi) = (IoU(Di, Pj) + DE(Di, Pj) + R(Di, Pj)) / 3,  for i ∈ D, j ∈ P
        """
        pass

    #Element-wise Weighted Mean of Every Cost Matrix Value (𝑊𝑀(𝐷,𝑃))
    def weighted_mean_cost_matrix(self, tracks, detections, image_dims, lambda_iou=0.33, lambda_de=0.33, lambda_r=0.34):
        """
        Computes the element-wise weighted mean of every cost matrix value:

        WM(Di, Pi) = (λ_IoU * IoU(Di, Pi) + λ_DE * DE(Di, Pi) + λ_R * R(Di, Pi)) / (λ_IoU + λ_DE + λ_R)

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
        num_tracks = cost_matrix.shape[0]
        num_detections = cost_matrix.shape[1]

        if num_tracks != (track_classes) or num_detections != len(detection_classes):
            raise ValueError("Dimensions of cost_matrix, track_classes, and detection_classes do not match - Class Gate Cost Matrix")

        #Create a boolean mask where classses match
        #             Reshapes to (num_tracks, 1)     Reshapes to (1, num_detections)
        match_mask = (track_classes[:, None] == detection_classes[None, :]) #Shape = [num_tracks, num_detections]
        #Because track_classes has the same number of columns as detection_classes has rows, we can perform matrix multiplication
        
        #Apply the mask and keep the values where classes match, zero where they do not
        gated_cost_matrix = cost_matrix * match_mask
        
        return gated_cost_matrix
