
#This file will consist of the YOLOv11 download and any pre-processing needed.

#YOLOv11 Import

#make a venv and pip install opencv-python ultralytics + lap
#import necessary libraries
#//from ultralytics import YOLO

#//model = YOLO("yolo11x.pt") #pt = pre-trained model
    #yolo11 has 5 versions from nano to x-large: yolo11(n, s, m, l, x)
    #yolo11n is the fastest, but least accurate

#Tracking with BoT-SORT (default yolo tracker)
    #//results = model.track(source="https://www.youtube.com/watch?v=XJse6aLQT8Y&ab_channel=NBCSports", conf=0.3, iou=0.5, show=True, save=True)
    #change source to any url or file/path

#Tracking with ByteTrack
    #//results = model.track(ssource="https://www.youtube.com/watch?v=XJse6aLQT8Y&ab_channel=NBCSports", tracker = "bytetrack.yaml", conf=0.20, iou=0.3, show=True, save=True)

#------------------------------------------------------------------------------------------------

#Python Script w/ OpenCV (cv2) and YOLOv11 for object tracking on video frames or live video feed
import cv2
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

#definition for video capture object
cap = cv2.VideoCapture("https://www.youtube.com/watch?v=XJse6aLQT8Y&ab_channel=NBCSports")

#loop through video frames
while True:
    ret, frame = cap.read()
    if ret:
        #run YOLOv11 tracking on frames
        results = model.track(frame, presist=True)
        #visualized results on frame
        annotated_frame = results[0].plot()
        #display annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break #break loop if q key is pressed
    else:
        break
cap.release()
cv2.destroyAllWindows()
