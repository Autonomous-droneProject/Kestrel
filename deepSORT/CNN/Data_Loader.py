import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Market1501(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.paths = [ file for file in os.listdir(image_dir) if file.endswith(".jpg") ]
        self.ids = set([ path.split("_")[0] for path in self.paths ])
        self.id_to_label = {pid: val for val, pid in enumerate(self.ids)}
        self.num_classes = len(self.id_to_label)

    def __getitem__(self, index):
        # Get name and path corresponding to index
        name = self.paths[index]
        path = os.path.join(self.image_dir, name)

        image = Image.open(path).convert("RGB")
        pid = name.split("_")[0] # gets id from image
        label = self.id_to_label[pid] # converts string id to clean int label

        return image, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.paths)
            
