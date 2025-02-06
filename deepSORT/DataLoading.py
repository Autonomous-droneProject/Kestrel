import pandas as pd
import torch
class DataLoading:
    def load_youtube_data(self, csv_path):
        '''
        This method will load data from the Youtube BoundingBoxes data set from google.
        We'll use the data from this dataset to help train and test the data association metric for deepSORT
        '''
        df = pd.read_csv(csv_path, header=None, names=["youtube_id", "timestamp_ms", "class_id", "class_name", "object_id", "object_presence", "xmin", "xmax", "ymin", "ymax"])
        tracking_data = {}

        #Assuming the videos are in 30 FPS, we want to normalize the timestamp to frame indexes since they're in ms


csv_path = "C:\Users\adamm\OneDrive\Documents\AI Projects\Kestrel\yt_bb_classification_validation.csv"
load_data = DataLoading()
tracking_data = load_data.load_youtube_data(csv_path)