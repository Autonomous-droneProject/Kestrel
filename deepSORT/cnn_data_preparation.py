import os
import torch
import configparser
#Python Imaging Library
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#I am training this CNN for Project Kestrel deepSORT using MOT challenge dataset

#1. Extract frames from the images in a directory

def extract_frames_from_images(image_dir):
    '''This function reads and loads image frames from a given directory

    Args:
        image_dir (string): path to the directory containing image frames.

    Output:
        List[PIL.Image]: list of loaded video frames
    '''

    video_frames = []
    image_files = sorted(os.listdir(image_dir)) #Sort to ensure the correct ordering of frames

    for image_file in image_files:
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path) #Load image using PIL
            video_frames.append(image)

    return video_frames

#2. Load sequence information from the .ini file, MOT challenge uses .ini files
def load_ini_file(ini_file):
    '''
    This function reads metadata from a sequence .ini file.

    Args:
        ini_file (string): Path to the .ini file

    Output:
        dict: n*2 array of sequence metadata
    '''

    config = configparser.ConfigParser()
    config.read(ini_file)

    sequence_info = {}
    sequence_info['name'] = config['Sequence']['name']
    sequence_info['imDir'] = config['Sequence']['imDir']
    sequence_info['frameRate'] = int(config['Sequence']['frameRate'])
    sequence_info['seqLength'] = int(config['Sequence']['seqLength'])
    sequence_info['imWidth'] = int(config['Sequence']['imWidth'])
    sequence_info['imHeight'] = int(config['Sequence']['imHeight'])
    sequence_info['imExt'] = config['Sequence']['imExt']

    return sequence_info

#3. Load object annotations from a CSV file
def load_annotations(annotation_file):
    '''
    This function loads object bounding box annotations.

    Args:
        annotation_file (string): path to annotation file

    Output:
        dict: Annotations stored in a dictionary format
    '''
    # Initialize ConfigParser
    config = configparser.ConfigParser()

    # Make sure to allow parsing the file even if sections aren't fully formatted
    config.optionxform = str  # Ensures that the case of keys is preserved

    try:
        # Read the .ini file (with option to ignore blank lines or comments)
        config.read(annotation_file)

        # Check if 'Sequence' section exists in the .ini file
        if 'Sequence' not in config.sections():
            print("Error: 'Sequence' section is missing from the .ini file.")
            return {}

        # Extract the values from the [Sequence] section
        sequence_info = {
            'name': config.get('Sequence', 'name'),
            'imDir': config.get('Sequence', 'imDir'),
            'frameRate': config.getint('Sequence', 'frameRate'),
            'seqLength': config.getint('Sequence', 'seqLength'),
            'imWidth': config.getint('Sequence', 'imWidth'),
            'imHeight': config.getint('Sequence', 'imHeight'),
            'imExt': config.get('Sequence', 'imExt')
        }

        # Return the extracted information
        return sequence_info

    except Exception as e:
        print(f"An error occurred while reading the .ini file: {e}")
        return {}


#4. Custom Dataset for DeepSORT tracking
class TrackingDataset(Dataset):
    def __init__(self, video_path, annotations, transform=None):
        """
        Custom dataset for loading and processing image frames

        Args:
            video_path (string): Path to image directory
            annotations (dict): Bounding box annotations
            transform (callable, optional): Transformations for images
        """
        self.video_path = video_path
        self.annotations = annotations
        self.transform = transform
        self.frame_files = sorted(os.listdir(video_path))  # Sorted list of images

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative) for training.

        Args:
            idx (int): Index of the frame.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Triplet of images.
        """
        frame_file = self.frame_files[idx]
        frame = Image.open(os.path.join(self.video_path, frame_file))

        frame_id = idx  # Assume frame index matches frame_id
        objects = self.annotations.get(frame_id, [])

        crops = []
        for obj in objects:
            object_id, box = obj
            cropped_object = frame.crop((box[0], box[1], box[2], box[3]))
            crops.append(cropped_object)

        if self.transform:
            crops = [self.transform(crop) for crop in crops]

        # Ensure we always return a valid triplet
        if len(crops) < 3:
            return crops[0] if crops else torch.zeros(3, 224, 224), \
                   crops[0] if crops else torch.zeros(3, 224, 224), \
                   crops[0] if crops else torch.zeros(3, 224, 224)

        return crops[0], crops[1], crops[2]

# 5. Collate function to filter None values
def collate_fn(batch):
    '''processes the list of samples from a batch'''
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None