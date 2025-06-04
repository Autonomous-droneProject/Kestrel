import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TrainingData(Dataset):
    def __init__(self, img_dirs, det_files, transform=None):
        """
        Args:
            img_dirs (list): List of image directories (one per sequence)
            det_files (list): List of detection files corresponding to img_dirs
            transform (callable, optional): Optional transform to apply to images
        """
        self.img_dirs = img_dirs
        self.det_files = det_files
        self.transform = transform
        self.detections = self.read_detections()  # Read detections for all files
        self.image_paths, self.labels = self.get_all_image_paths_and_labels()  # Gather images and labels

    def read_detections(self):
        """
        Reads detection data from multiple MOT16 detection files.
        Returns:
            Dictionary mapping (frame_id, img_dir) -> list of (object_id, bbox)
        """
        detections = {}

        for img_dir, det_file in zip(self.img_dirs, self.det_files):
            if not os.path.exists(det_file):
                print(f"Warning: Detection file {det_file} not found, skipping.")
                continue  # Skip missing files

            with open(det_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame_id = int(parts[0])  # Frame ID
                    object_id = int(parts[1])  # Object ID (LABEL)

                    if object_id == -1:  # Ignore unknown object IDs
                        continue

                    bbox = list(map(float, parts[2:6]))  # Bounding Box (x, y, w, h)
                    key = (frame_id, img_dir)

                    if key not in detections:
                        detections[key] = []
                    detections[key].append((object_id, bbox))

        print(f"Total valid detections loaded: {sum(len(v) for v in detections.values())}")
        return detections

    def get_all_image_paths_and_labels(self):
        """
        Collects all image paths and their corresponding object labels.
        Returns:
            List of image paths and their associated object IDs (labels).
        """
        image_paths = []
        labels = []

        for img_dir in self.img_dirs:
            if not os.path.exists(img_dir):
                print(f"Warning: Image directory {img_dir} not found, skipping.")
                continue

        image_files = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        )

        print(f"Found {len(image_files)} images in {img_dir}")

        for image_file in image_files:
            frame_id = int(os.path.basename(image_file).split('.')[0])  # Extract frame ID from filename
            key = (frame_id, img_dir)

            if key in self.detections:
                for obj_id, bbox in self.detections[key]:
                    image_paths.append(image_file)
                    labels.append(obj_id)  # Store the object ID as the label

        print(f"Total valid images found: {len(image_paths)}")
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]  # Retrieve the corresponding object ID

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping.")
            return None, None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load {image_path}, skipping.")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label  # Return both image and object ID

# Custom collate function to remove None values
def custom_collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]  # Remove None values
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)  # Return empty tensors if no valid samples
    return torch.utils.data.dataloader.default_collate(batch)

# MOT16 Dataset File Paths
img_dirs = [
    r'C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\MOT16\train\MOT16-02\img1',
    r'C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\MOT16\train\MOT16-04\img1',
    r'C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\MOT16\train\MOT16-05\img1',
]

det_files = [
     r'C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\MOT16\train\MOT16-02\det\det.txt',
     r'C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\MOT16\train\MOT16-04\det\det.txt',
     r'C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\MOT16\train\MOT16-05\det\det.txt',
]

# Define Image Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset and DataLoader
train_dataset = TrainingData(img_dirs, det_files, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
