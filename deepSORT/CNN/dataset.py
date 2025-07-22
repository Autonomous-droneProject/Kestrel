import os
import torch
import random
from torch.utils.data import Dataset  # The base class we will inherit from to create our custom dataset
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from PIL import Image  # Python Imaging Library (Pillow), used for opening and manipulating image files
import pickle
import copy

class Market1501(Dataset):
    """
    Custom PyTorch Dataset for the Market-1501 dataset.

    This class handles loading images and parsing their person IDs (PIDs)
    from the filenames to be used as integer labels for training a classification model.
    """

    # CHANGED: Added 'transform=None' to accept an image transformation pipeline.
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory path with all the training images.
            transform (callable, optional): A function/transform to be applied to each image.
        """
        # Store the directory path for later use in __getitem__.
        self.image_dir = image_dir

        # Store the transform pipeline (e.g., resize, normalize) for later use.
        self.transform = transform

        # Create a list of all filenames in the directory that end with '.jpg'.
        # This is a 'list comprehension', a compact way to build a list.
        # It filters out any non-JPEG files (like '.txt' or system files) that might cause errors.
        self.image_paths = [file for file in os.listdir(image_dir) if file.endswith(".jpg")]

        # Create a set of unique Person IDs (PIDs) from the filenames.
        # Example filename: '0002_c1s1_000451_01.jpg'. We split by '_' and take the first part '0002'.
        # Using a 'set' automatically removes all duplicate PIDs, giving us a list of unique people.
        pids_in_dataset = set([path.split("_")[0] for path in self.image_paths])

        # Create a mapping dictionary to convert string PIDs to integer labels (0, 1, 2, ...).
        # Neural networks and loss functions require integer labels, not strings.
        # 'enumerate' provides a counter 'i' for each unique 'pid'.
        # Example: {'0002': 0, '0007': 1, '-1': 2}
        self.pid_to_label = {pid: i for i, pid in enumerate(pids_in_dataset)}

        # Store the total number of unique classes (people).
        # This is needed to correctly configure the output layer of our neural network.
        self.num_classes = len(self.pid_to_label)

    # The __getitem__ method defines how to retrieve a single sample (one image and its label) from the dataset.
    # The DataLoader will call this method automatically for each index from 0 to len(dataset)-1.
    def __getitem__(self, index):
        # Retrieve the filename for the given index from our list of paths.
        name = self.image_paths[index]

        # Construct the full, platform-independent path to the image file.
        # os.path.join is used so this code works on both Windows ('\') and Linux/Mac ('/').
        path = os.path.join(self.image_dir, name)

        # Open the image file using Pillow and ensure it's in RGB format (corresponds to 3 channels).
        # This is important because some images might be in grayscale or have an alpha channel.
        image = Image.open(path).convert("RGB")

        # CHANGED: Apply the transformations to the image.
        # This is a CRITICAL step. It will resize, normalize, and convert the PIL Image to a PyTorch Tensor.
        if self.transform:
            image = self.transform(image)

        # Parse the filename again to get the string Person ID.
        pid = name.split("_")[0]

        # Use our mapping dictionary to look up the integer label for the corresponding PID.
        label = self.pid_to_label[pid]

        # Return the processed image tensor and its corresponding label as a tensor.
        # The label must be a LongTensor (64-bit integer) for PyTorch's CrossEntropyLoss function.
        return image, torch.tensor(label, dtype=torch.long)
    
    # The __len__ method must return the total number of samples in the dataset.
    # The DataLoader needs this to know the dataset's size for batching, shuffling, and epoch completion.
    def __len__(self):
        # Return the total count of the image paths we found during initialization.
        return len(self.image_paths)
            
class AG_VPReID(Dataset):
    def __init__(self, root_dir, transform=None, cache_file = "AG_VPReID.pkl", samples=None):
        self.samples = [] # (frame_path, label)
        self.transform = transform
        self.cache_file = os.path.join(root_dir, cache_file)

        if samples is not None:
            # 1. Use pre-filtered samples if provided (for train/val splits)
            print("[Cache] Loading pre-filtered samples.")
            self.samples = samples
        elif os.path.exists(self.cache_file):
            # 2. If no samples are provided, try loading from the cache file
            print(f"[Cache] Loading samples from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                self.samples = pickle.load(f)
        else:
            # 3. If no cache exists, build the sample list from scratch and save it
            print("[Cache] Building sample list from scratch...")
            self._load_paths(root_dir)
            print(f"[Cache] Saving to {self.cache_file}")
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.samples, f)

    
    def _load_paths(self, root_dir): #The underscore in this method name is just a convention to show that it is only meant to be called internally, not from objects of this class
        #All directories within the root
        directories_list = os.listdir(root_dir)

        #For all images within the list of directories
        for person_ID in sorted(directories_list):
            try:
                #Build a full path to the image directory
                person_path = os.path.join(root_dir, person_ID)
                if not os.path.isdir(person_path):
                    continue
                for tracklet in os.listdir(person_path):
                    #Build a full path to each tracklet
                    tracklet_path = os.path.join(person_path, tracklet)

                    for frame in os.listdir(tracklet_path):
                        #Build a full path to each frame
                        frame_path = os.path.join(tracklet_path, frame)
                        self.samples.append((frame_path, int(person_ID))) #Store the person ID as an int So now we have each image and ID, ! will fail if person_ID is not a number
            except (ValueError, NotADirectoryError):
                # Gracefully skip this directory if it's not in the expected format
                print(f"Warning: Skipping invalid directory or contents: {person_path}")
                continue
    
    def __getitem__(self, index):
        #Extract the values from the list of tuples
        frame_path, label = self.samples[index]
        #Open the image and convert it to RGB
        image = Image.open(frame_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype = torch.long)

            
    def __len__(self):
        return len(self.samples)

# Ensures that every batch has a rich structure of positive and negative pairs, making the loss calculation more stable and meaningful.
class PKsampler(Sampler):
    """
    P-K Sampler. It creates batches by first sampling P identities,
    then sampling K instances for each of those identities.
    Fix: Partitions the dataset into disjoint batches of P identities and K instances.
    
    Args:
        dataset: The dataset to sample from. Must have a 'samples' attribute
                 where each element is a tuple like (path, pid).
        p: Number of distinct identities per batch.
        k: Number of instances per identity per batch.
    """
    def __init__(self, dataset, p, k):
        super().__init__(dataset)
        self.p = p
        self.k = k
        
        # Group indices by person ID
        self.pid_to_indices = defaultdict(list)
        for index, (_, pid) in enumerate(dataset.samples):
            self.pid_to_indices[pid].append(index)
            
        self.pids = list(self.pid_to_indices.keys())
        
        # Calculate the number of batches
        # This is the number of groups of P identities we can form
        self.num_batches = len(self.pids) // self.p

    def __iter__(self):
        # Shuffle the person IDs for this epoch
        pids_for_epoch = random.sample(self.pids, len(self.pids))
        
        # Yield batches
        for i in range(self.num_batches):
            start_idx = i * self.p
            end_idx = start_idx + self.p
            batch_pids = pids_for_epoch[start_idx:end_idx]
            
            batch_indices = []
            for pid in batch_pids:
                pid_indices = self.pid_to_indices[pid]
                
                # Sample K instances for this PID
                if len(pid_indices) >= self.k:
                    sampled_indices = random.sample(pid_indices, self.k)
                else:
                    sampled_indices = random.choices(pid_indices, k=self.k)
                
                batch_indices.extend(sampled_indices)
                
            yield batch_indices

    def __len__(self):
        # Approximate number of batches per epoch
        return self.num_batches
    
if __name__ == '__main__':
    # This code is now safe and only runs for testing this file
    root_dir = r'B:\Downloads\train'
    dataset = AG_VPReID(root_dir)
    print(f"Loaded {len(dataset)} samples.")


