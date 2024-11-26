#You need to add the sharedModules directoy to the python path

from sharedModules.telloPreProcessing import PreProccessing as TpreP
from sharedModules.YOLO.yoloHandler import YOLOModelHandler
from sharedModules.telloPostProcessing import PostProcessing as TpostP
from SORT.DataAssociation import DataAssociation as DAM
from sharedModules.KalmanFilter import KalmanFilter as KF
from sharedModules.TrackManagement import TrackManagement as TM


# Creating objects for each class
tello_preprocessing = TpreP()       # PreProcessing class
yolo_handler = YOLOModelHandler()   # YOLOModelHandler class
tello_postprocessing = TpostP()     # PostProcessing class
data_association = DAM()            # DataAssociation class
kalman_filter = KF()                # KalmanFilter class
track_management = TM()             # TrackManagement class


print("HELLO")
