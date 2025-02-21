import os
import torch
import cv2
from ultralytics import YOLO
from CNNDeepSort import CNNDeepSORT
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Paths and Model Loading
# --- Paths and Model Loading ---
BASE_DIR = r"C:\Users\adamm\PycharmProjects\Kestrel\deepSORT"
CNN_PATH = os.path.join(BASE_DIR, "CNN", "deepsort_cnn.pth")
IMAGE_DIR = r"C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\test\MOT17-01\img1"
YOLO_MODEL_PATH = "yolov8n.pt"


#Load the custom-made CNN
def load_cnn_model(model_path):
    model = CNNDeepSORT()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

cnn_model = load_cnn_model(CNN_PATH)
cnn_model.eval()
yolo_model = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = yolo_model.names


'''
#Function to convert image to tensor for CNN feature extraction
#Adjust to 128D embedding MOT standard
#Deep NN typically work with floats, images are typically initially ints
#Channels first, images are typically "channels last" where the channels are the last dimension
#The line above changes the img dimensions by swapping the indices, from (height, width, channels) to (channels, height, width). Which is what PyTorch expects
#Add a batch dimension to the tensor at position 0
'''
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))
    img=torch.tensor(img).unsqueeze(0)
    return img

def run_object_detection_yolo(frame, model):
    results = model(frame)
    detections = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box.tolist()) #Maps variable to a list
            class_id = int(cls)
            detections.append((x_min, y_min, x_max, y_max, class_id))
    return detections

#implement the CNN feature extractor by comparing embeddings to see if the reappearing object is the same object that disappeared.
#We'll do this with cosine similarity.
'''
    
'''
def compare_embeddings(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.reshape(1,-1), embedding2.reshape(1,-1))[0][0]
    return similarity

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image: {image_path}")
        continue

    detections = run_object_detection_yolo(frame, yolo_model)

    embeddings = [] #Store the embeddings for the current image

    for x_min, y_min, x_max, y_max, class_id in detections:
        cropped_img = frame[y_min:y_max, x_min:x_max]
        preprocessed_img = preprocess_image(cropped_img)

        with torch.no_grad():
            embedding = cnn_model(preprocessed_img)
            embedding = embedding.cpu().numpy()
        embeddings.append(embedding)

        # Draw bounding boxes and labels (optional, for visualization)
        label = f"{CLASS_NAMES[class_id]}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text_x, text_y = x_min, max(y_min - 10, 20)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (text_x, text_y - h - 5), (text_x + w, text_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    #Compare embeddings with all pairs within the image
