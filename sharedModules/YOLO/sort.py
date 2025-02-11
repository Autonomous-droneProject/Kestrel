from filterpy.kalman import KalmanFilter
import numpy as np
from math import sqrt, pow 
import scipy
import scipy.optimize

def cost_matrix(dect: np.ndarray, pred: np.ndarray):
    """
    This functions takes in a numpy array of detections from the YOLO models
    and a list of predictions from the kalman filter tracker, and returns a iou cost
    matrix.
    """

    # avoid broadcasting issues
    dect = np.expand_dims(dect, axis=1) 
    pred = np.expand_dims(pred, axis=0) 

    # get two points from each detected bbox
    x1D = dect[..., 0] - dect[..., 2]/2
    y1D = dect[..., 1] - dect[..., 3]/2
    x2D = dect[..., 0] + dect[..., 2]/2
    y2D = dect[..., 1] + dect[..., 3]/2

    # get two points from each predicted bbox
    x1P = pred[..., 0] - pred[..., 2]/2
    y1P = pred[..., 1] - pred[..., 3]/2
    x2P = pred[..., 0] + pred[..., 2]/2
    y2P = pred[..., 1] + pred[..., 3]/2

    # choose the maximum bewteen the two points
    x1 = np.maximum(x1D, x1P)
    y1 = np.maximum(y1D, y1P)
    x2 = np.minimum(x2D, x2P)
    y2 = np.minimum(y2D, y2P)
    intersection_width = np.maximum(0, x2-x1)
    intersection_height = np.maximum(0, y2-y1)
    area_of_intersection = intersection_height * intersection_width
    
    dect_area = dect[..., 3] * dect[..., 2]
    pred_area = pred[..., 3] * pred[..., 2]
    area_of_union = dect_area + pred_area - area_of_intersection
    
    iou_matrix = np.where(area_of_union > 0, area_of_intersection / area_of_union, 0.0)

    return iou_matrix

def linear_assignment(costMatrix: np.float64):
    """
    This function will take in a cost matrix and return an array of matched
    indeces for the rows and columns of the matrix throught linear assignment
    """
    # do a linear assignment based on the cost matrix 
    # the return is a list of row and column indexes with the optimal assignment
    dect, pred = scipy.optimize.linear_sum_assignment(costMatrix, True)
    # make an array of indexes of detections matched with indexes of predictions
    match_indxs = np.array(list(zip(dect,pred)))

    return match_indxs


def convert_xywh_to_xyxy(bbox: np.int32):
    x1 = bbox[0] - bbox[2]/2
    x2 = bbox[0] + bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    y2 = bbox[1] + bbox[3]/2
    z = np.array([x1,y1,x2,y2]).astype(int)
    return z

def convert_xyxy_to_xywh(bbox: np.int32):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[2] - w/2
    y = bbox[3] - h/2
    z = np.array([x,y,w,h]).astype(int)
    return z

def convert_bbox_to_z(bbox: np.int32):
    w=bbox[2] 
    h=bbox[3]
    area = w*h
    aspect_ratio = w/h
    z = np.array([bbox[0], bbox[1], area, aspect_ratio])
    return z

def convert_z_to_bbox(bbox: np.int32):
    w = int(sqrt(abs(bbox[2]*bbox[3])))
    h = int(sqrt(abs( bbox[2]*pow(bbox[3], -1) )))
    new_bbox = np.array([bbox[0], bbox[1], w, h])
    return new_bbox

class Kfilter():
    def __init__(self):
        

        # create a kalman filter instance
        self.dim_x = 7
        self.kalman = KalmanFilter(dim_x=self.dim_x, dim_z=4)

        #state transition matrix
        self.kalman.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 1]])
        
        # measurement function
        self.kalman.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0]])

        #covariance matrix, P already contains np.eye(dim_x)
        self.kalman.P *= 1000.

        # measurement noise
        self.kalman.R = np.eye(4)
        self.kalman.R *= 10.

        # assign the process noise
        self.kalman.Q = np.eye(self.dim_x)
        self.kalman.Q *= 0.01

        #set arbitrary initial value
        self.kalman.x = np.array([0, 0, 0, 0, 0, 0, 0])

    # make a prediction
    def predict(self):
        self.kalman.predict()
        bbox = convert_z_to_bbox(self.kalman.x)

        return bbox
    
    def update(self, bbox):
        bbox = convert_bbox_to_z(bbox)
        self.kalman.update(bbox)

class KalmanTracker():
    def __init__(self):
        self.tracks = list()
        self.id = 0
    
    def create_track(self, bbox):
        track = {
            "id": self.id,
            "filter": Kfilter(),
            "history": [bbox],
        }
        self.id += 1
        return track

    def associate_detections(self, preds: list, dects: np.int32, iou_threshold = 0.3):
        """
        This function is in charged of associating the detections matched by the linear
        sum assignment to their corresponding track, and return a list of matched bboxes,
        unmatched bboxes, and unmatched tracks to be deleted
        """


        iou_matrix = cost_matrix(dects, preds)
        matched_indexes = linear_assignment(iou_matrix)

        # get unmatched bounding boxes list from the detections by comparing
        # if any of the indexes in the detections array are not in the 
        # matched indeces array
        unmatched_bboxes = list()
        for i, dect in enumerate(dects):
            if i not in matched_indexes[:, 0]:
                unmatched_bboxes.append(i)
        
        # similarly, get the unmatched tracks
        unmatched_tracks = list()
        for j, pred in enumerate(preds):
            if j not in matched_indexes[:, 1]:
                unmatched_tracks.append(j)

        # get the matched detections and filter out the low ious
        matched_detections = list()
        for matched in matched_indexes:
            # if the detection doesn't pass the threshold
            if (iou_matrix[matched[0], matched[1]] < iou_threshold):
                # append the index of the detection to the unmatched detection list
                unmatched_bboxes.append(matched[0])
                # append the index of the prediction to the unmatched track list
                unmatched_tracks.append(matched[1])
            
            else:
                # append it to the matched detections list
                matched_detections.append(matched)
        
        return matched_detections, unmatched_bboxes, unmatched_tracks

    def pred_tracks(self):
        lstOfPred = list()
        if len(self.tracks) == 0:
            return
        
        for track in self.tracks:
            # if the length of the track is just 1, then initialize the kalman filter with that value
            if len(track["history"]) == 1:
                track["filter"].kalman.x = np.append(convert_bbox_to_z(track["history"][-1]), [0,0,0])
                # predict the kalman filter, get the new bounding box
                nbbox = track["filter"].predict()
            else:
                nbbox = track["filter"].predict()
            
            lstOfPred.append(nbbox)
        
        return lstOfPred
    
    def update_tracks(self, matched_indexes: list, bboxes: np.ndarray):
        for m in matched_indexes:
            # add on the corresponding dectections to each track's history
            self.tracks[m[1]]["history"].append(bboxes[m[0]])
            # update each matched track filter
            self.tracks[m[1]]["filter"].update(self.tracks[m[1]]["history"][-1])

class Sort(KalmanTracker):
    def __init__(self):
        super().__init__()
        self.bboxes = np.empty((2,3))

    def get_bboxes(self, bboxes: np.ndarray):
        self.bboxes = bboxes

    def manage_tracks(self, bboxes: np.ndarray):
        """
        This function is in charged of managing all the tracks correctly and returns
        a list of tracks
        """
        # get new bounding boxes
        self.get_bboxes(bboxes)

        # check to see if there are no tracks
        if (len(self.tracks) == 0):
            for bbox in self.bboxes:
                self.tracks.append(self.create_track(bbox))
        else:

            # do a prediction using the kalman filter
            list_predictions = self.pred_tracks()

            # get list of matched detections, unmatched detections, and unmatched tracks
            matches, unmatched_detections, unmatched_tracks = self.associate_detections(list_predictions, bboxes)

            # update kalman filter for each track
            self.update_tracks(matches, bboxes)

            # if a detection was not associated to any track
            # create a new track for it
            if len(unmatched_detections) > 0:
                for m in unmatched_detections:
                    self.tracks.append(self.create_track(bboxes[m]))
            
            # if there is a track that was not matched to any
            # detection, delete it
            if len(unmatched_tracks) > 0:
                for t in unmatched_tracks:
                    if t > len(self.tracks) - 1: continue
                    self.tracks.pop(t)
            
            

        return self.tracks









    