import cv2
import torch
import numpy as np

def extract_person_embedding(frame: np.ndarray, box, cnn_model) -> np.ndarray | None:
    """
    Crops a person from the frame, runs pre-processing, and extracts
    the appearance embedding using the CNN model.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    person_crop = frame[y1:y2, x1:x2]

    if person_crop.size == 0:
        return None

    # Pre-process the crop for the CNN
    crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB) # get rgb instead of bgr
    crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Run CNN inference to get the appearance vector
    with torch.no_grad():
        appearance_vector = cnn_model(crop_tensor)
        if hasattr(appearance_vector, "detach"):
            appearance_vector = appearance_vector.detach().cpu().numpy()
        
        # Squeeze the batch dimension out
        return np.squeeze(appearance_vector)