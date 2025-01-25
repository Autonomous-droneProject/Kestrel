import cv2
import time
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from math import sqrt, pow 

def iou_calculator(bb_test: np.int32, bb_gt: np.int32):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    # get the length and width of the box of the intersection
    x1 = max(bb_test[0], bb_gt[0]) 
    y1 = max(bb_test[1], bb_gt[1])
    x2 = min(bb_test[2], bb_gt[2])
    y2 = min(bb_test[3], bb_gt[3])
    intersection_width = max(0, x2-x1)
    intersection_height =  max(0, y2-y1)
    area_of_intersection = intersection_height * intersection_width
    #get the area of the ground truth
    gt_bb_len = abs(bb_gt[2] - bb_gt[0])
    gt_bb_width = abs(bb_gt[3] - bb_gt[1])
    gt_bb_area = gt_bb_len * gt_bb_width
    #get the area of the prediction
    det_bb_len = abs(bb_test[2] - bb_test[0])
    det_bb_width = abs(bb_test[3] - bb_test[1])
    det_bb_area = det_bb_len * det_bb_width
    #get area of the union
    area_of_union = gt_bb_area + det_bb_area - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou

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
                                # center of bounding box
        # self.kalman.x = np.array([self.bbox[0], self.bbox[1], 
        #                           # area and aspect ratio
        #                           self.bbox[2]*self.bbox[3], self.bbox[2]/self.bbox[3],
        #                           # change in position over time of center and area
        #                           0, 0, 0])
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
    
    def bboxes_to_tracks(self, bboxes: np.int32):
        """
        This function must handle:
            - creating tracks
            - extending tracks
                - ensure no two tracks have the same bounding box
                - iou calculation
            - eliminating uncessary tracks
            - controling tracks' size"""

        # if tracks is 0, then we must be at the first frame
        if len(self.tracks) == 0:
            for bbox in bboxes:
                self.tracks.append([bbox])
            
            return self.tracks
        
        convBBoxes = [convert_xywh_to_xyxy(bbox) for bbox in bboxes]
        convTrkedObj = [convert_xywh_to_xyxy(track[-1]) for track in self.tracks]
        #print(f"Track Obj: {convTrkedObj}")

        # if there are more tracks in the list than bounding boxes, we have to delete some tracks
        if len(self.tracks) > len(bboxes):
            # find how many tracks are extra
            exTrack = len(self.tracks) - len(bboxes)
            iouLst = list()
            # Calculate each track's iou with each bounding box, find the maximum
            for track in convTrkedObj:
                maxIoU = 0
                for bbox in convBBoxes:
                    iou = iou_calculator(track, bbox) 
                    if iou > maxIoU:
                        maxIoU = iou
                # store max in a list
                iouLst.append(maxIoU)
            # NOTE: the id of the max value will correspond with the id of the track
            # delete the tracks with the lowests IoUs
            trkEnumSorted = sorted(enumerate(iouLst), key= lambda x: x[1])
            self.tracks = [self.tracks[i] for i, _ in trkEnumSorted[exTrack:]]
            return self.tracks
        # elif there are less tracks in the list than bounding boxes, we have to create some tracks
        elif len(self.tracks) < len(bboxes):
            # find how many tracks are needed
            nTracks = len(bboxes) - len(self.tracks)
            iouLst = list()
            # Calculate each bounding box's iou with each track, find the maximum
            for bbox in convBBoxes:
                maxIoU = 0
                for track in convTrkedObj:
                    iou = iou_calculator(bbox, track) 
                    if iou > maxIoU:
                        maxIoU = iou
                # store max in a list
                iouLst.append(maxIoU)
            # NOTE: the id of the max value will correspond with the id of the bounding box
            # append to the tracks list new tracks with the bounding boxes with the lowest IoUs
            iouLow = sorted(enumerate(iouLst), key= lambda x: x[1])[:nTracks]
            #print(f"iouLow: {iouLow}")
            self.tracks.extend([[bboxes[i]] for i, _ in iouLow])
            return self.tracks

        # if the number of tracks equals the number of bounding boxes, then we don't need more tracks
        # Calculate the track's iou with each bounding box, store it in a list
        used = set()
        for _, track in enumerate(self.tracks):
            lasttrkobj = convert_xywh_to_xyxy(track[-1])
            maxIoU = 0
            idx = -1
            for i, bbox in enumerate(convBBoxes):
                if i in used:
                    continue
                iou = iou_calculator(lasttrkobj, bbox) 
                if iou > maxIoU:
                    # get the max value
                    maxIoU = iou
                    # find the index
                    idx = i
                    
            # append the bounding box to the corresponding track
            if idx != -1:
                track.append(bboxes[idx])
                used.add(idx)
            # delete the track's previous bounding box
            #print(len(track))
            max_track_size = 3
            if len(track) > max_track_size:
                # keep the last three delete the rest
                track = track[-max_track_size:]
        # return the tracks
        return self.tracks

    def pred_tracks(self):
        predBBoxesForTracks = list()
        for track in self.tracks:
            # predict the next bounding box in the track
            kfilter = Kfilter()
            # if the length of the track is just 1, then initialize the kalman filter with that value
            if len(track) == 1:
                kfilter.kalman.x = np.append(convert_bbox_to_z(track[-1]), [0,0,0])
                # predict the kalman filter, get the new bounding box
                nbbox = kfilter.predict()
                #update the kalman filter
                kfilter.update(track[-1])
            else:
            # the kalman filter object will get reinitialized every time this loop runs
            # therefore, on a hypothetical second call to this method, the original kalman filter for that
            # track will be lost. Hence, we must store the predictions of the kalman filter on the tracks list
            # so that when the method gets called again, it will be updated with all the values of the kalman filter
                for bbox in track:
                    kfilter.update(bbox)
                    nbbox = kfilter.predict()
            
            predBBoxesForTracks.append(nbbox)
                    
            

        return predBBoxesForTracks

