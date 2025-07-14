from ultralytics import YOLO
import logging
from CNN.model import CNNdeepSORT
import cv2
import torch
import numpy as np
import time

YOLOModel = YOLO('yolo11m.pt')
logging.getLogger("ultralytics").setLevel(logging.WARNING)

CNNModel = CNNdeepSORT(embedding_dim=128, num_classes=751)
cap = cv2.VideoCapture(0)

last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = YOLOModel(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if YOLOModel.names[cls_id] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            # Gets cropped images to send to CNN
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            appearance_vector = np.random.rand(128) # delete this and uncomment lines below to actually get appearance vector
            # with torch.no_grad():
            #     appearance_vector = CNNModel(crop_tensor)
            # turns this into np array to make it readable when printed
            #     if hasattr(appearance_vector, "detach"):
            #         appearance_vector = appearance_vector.detach().cpu().numpy()
            #     appearance_vector = np.squeeze(appearance_vector)

            if time.time() - last_print_time > 10:
                print(f"Appearance vector person at ({x1}, {y1}, {x2}, {y2}): \n{appearance_vector}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Person {conf: .2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if time.time() - last_print_time > 10:
        last_print_time = time.time()

    cv2.imshow('YOLOv8 Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()