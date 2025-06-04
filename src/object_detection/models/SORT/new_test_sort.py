import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np
from sklearn.cluster import KMeans

trt_model = YOLO('yolo11s.pt')

# Open the video file
video_path = 'people.avi'
cap = cv2.VideoCapture(video_path)


# get the video's dimmension
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
fps = cap.get(cv2.CAP_PROP_FPS)
   
size = (frame_width, frame_height) 

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('stream.avi', fourcc, 20, size)


fc = 0
sort = Sort()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fc += 1
    

    #initialization sequence
    if (fc == 1):
        blurred_frame = cv2.GaussianBlur(frame, (45, 45), 0)
        point_of_ref = (frame_width//2, frame_height//2)
        mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        mask = cv2.circle(mask, point_of_ref, 150, color=(255, 255, 255), thickness=-1)
        frame = np.where(mask==np.array([255, 255, 255]), frame, blurred_frame)
    # for each subsequent frame
    else:
        blurred_frame = cv2.GaussianBlur(frame, (45, 45), 0)
        mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        mask = cv2.circle(mask, point_of_ref, 150, color=(255, 255, 255), thickness=-1)
        frame = np.where(mask==np.array([255, 255, 255]), frame, blurred_frame)
        

    # Run inference
    results = trt_model.predict(frame)
    # Process results
    for result in results:
        # process results as a dataframe (convinience)
        df = result.to_df()
        # filter out objects based on class and confidence score
        df.drop(df[(df['class'] != 0) | (df['confidence'] < .6)].index , inplace=True)
        # get a new array of bounding boxes
        bboxes = np.array([result.boxes.xywh.cpu().numpy().astype(int)[i] for i in df.index.values])
        # get the coordinates of each bbox
        positions = np.array([(x, y) for x, y, _, _, in bboxes])

        try:
            # do a kmeans clustering with only 1 cluster
            kmeans = KMeans(n_clusters=1, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(positions).tolist()
            # get the center of that cluster
            centroid = kmeans.cluster_centers_[labels[0]]
            # update our point_of_reference (center of the circle)
            point_of_ref = tuple(centroid.astype(int).tolist())
        # if there are no objects to cluster
        except ValueError:
            # set the point of reference to the center of the screen
            point_of_ref = (frame_width//2, frame_height//2)

        # pass the bboxes to SORT
        try: 
            tracks = sort.manage_tracks(bboxes, 0.5)
        # go to the next frame if there aren't any
        except:
            break

        # draw a bounding box for each track    
        for track in tracks:
            nbbox = track["history"][-1]
            x1 = int(nbbox[0] - nbbox[2]/2)
            y1 = int(nbbox[1] - nbbox[3]/2)
            x2 = int(nbbox[0] + nbbox[2]/2)
            y2 = int(nbbox[1] + nbbox[3]/2)
            cv2.putText(
                frame,
                str(track["id"]),
                (int(x1), int(y1) - 10),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (0, 255, 0),
                thickness=2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the frame to the VideoWriter object
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


out.release()
cap.release()
cv2.destroyAllWindows()