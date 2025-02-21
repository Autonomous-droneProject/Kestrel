import torch
import cv2
import numpy as np
from PIL import Image

from deepSORT.CNN.CNNDeepSort import CNNDeepSORT

#Load the trained CNN
model = CNNDeepSORT()
model.load_state_dict(torch.load("deepsort_cnn.pth"))
model.eval()

#Image preprocessing
def preprocess_image(image_path):
