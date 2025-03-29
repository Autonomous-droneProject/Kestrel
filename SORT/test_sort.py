import cv2
from ultralytics import YOLO
from sort import Sort

trt_model = YOLO('yolo11m.pt')

# Open the video file
video_path = 'cars4.avi'
cap = cv2.VideoCapture(video_path)


# We need to set resolutions. 
# so, convert them from float to integer. 
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

    # Run inference
    # results = yolo_pipeline(images=frame)
    results = trt_model.predict(frame)

    # Process results (similar to your existing code)
    for result in results:
        bboxes = result.boxes.xywh.cpu().numpy().astype(int)
        #if fc == 1: predBBoxes = KalmanTracker()
        #tracks = predBBoxes.bboxes_to_tracks(bboxes)
        tracks = sort.manage_tracks(bboxes)

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