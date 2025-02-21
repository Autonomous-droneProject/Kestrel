#This script is for testing the CNN for Project Kestrel
import configparser
import os
import numpy as np
import torch
from CNNDeepSort import CNNDeepSORT
import cv2
from ultralytics import YOLO

# Paths (Adjust as needed)
BASE_DIR = r"/deepSORT"
CNN_PATH = os.path.join(BASE_DIR, "CNN", "deepsort_cnn.pth")
INI_PATH = os.path.join(BASE_DIR, "test", "MOT17-01", "seqinfo.ini")

def load_cnn_model(model_path):
    # Load the model
    model = CNNDeepSORT()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

#Parse the annotations file

config = configparser.ConfigParser()
config.read(INI_PATH)

#Extract sequence information from annotations file
image_dir = os.path.join(os.path.dirname(INI_PATH), config["Sequence"]["imDir"])
frame_rate = int(config["Sequence"]["frameRate"])
seq_length = int(config["Sequence"]["seqLength"])
img_width = int(config["Sequence"]["imWidth"])
img_height = int(config["Sequence"]["imHeight"])
img_ext = config["Sequence"]["imExt"]

#Get a sorted list of image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(img_ext)])

#Output video setup
output_path = "tracked_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (img_width, img_height))

#Function to convert image to tensor for CNN feature extraction
def preprocess_image(img):
    img = cv2.resize(img, (128, 128)) #Adjust to 128D embedding MOT standard
    img = img.astype(np.float32) / 255. #Deep NN typically work with floats, images are typically initially ints
    img = np.transpose(img, (2, 0, 1))  # Channels first, images are typically "channels last" where the channels are the last dimension
    #The line above changes the img dimensions by swapping the indices, from (height, width, channels) to (channels, height, width). Which is what PyTorch expects
    img = torch.tensor(img).unsqueeze(0) #Add a batch dimension to the tensor at position 0
    return img


#Run YOLO object detection on a frame to create a detection
def run_object_detection_YOLO(frame, model):
    results = model(frame)
    detections = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            class_id = int(cls) #Class index
            detections.append((int(x_min), int(y_min), int(x_max), int(y_max), class_id))

    return detections

cnn_model = load_cnn_model(CNN_PATH)
yolo_model = YOLO("yolov8n.pt")
CLASS_NAMES = yolo_model.names #Get class names from the model

#Process each frame

for file in image_files:
    frame_path = os.path.join(image_dir, file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Skipping {frame_path}, could not read file.")
        continue

    detections = run_object_detection_YOLO(frame, yolo_model)

    #Draw bounding boxes and class id
    for (x_min, y_min,x_max, y_max, class_id) in detections:
        #Draw the rectangle around the detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        #Prepare label text (class id)
        label = f"{CLASS_NAMES[class_id]}"

        #Position for label text
        text_x, text_y = x_min, max(y_min - 10, 20) #above the bounding box

        #Draw the text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (text_x, text_y - h - 5), (text_x + w, text_y + 5), (0, 255, 0), -1)

        # Put text on the frame
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    out.write(frame) #Write frame to the video

out.release()
print("Completed!, output is saved in: ", output_path)
