class YOLOModelHandler:

    def __init__(self, model_path: str, confidence: float = 0.3):
        """
        Constructor for the YOLO model handler.
            model_path (str): Path to the YOLO model file.
            confidence (float): Confidence threshold for detections. Default is 0.5.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def train_model(self, data_path: str, epochs: int = 100, img_size: int = 640):
        """
        Trains the YOLO model on a specified dataset.
            data_path (str): Path to the dataset YAML file.
            epochs (int): Number of epochs to train. Default is 100.
            img_size (int): Image size for training. Default is 640.
        """
        results = self.model.train(data=data_path, epochs=epochs, imgsz=img_size)
        return results

    def detect(self, video_path: str, save: bool = True, show: bool = True, classes: list = None):
        """
        Runs object detection and returns results.
            video_path (str): Path to the video file.
            save (bool): Whether to save the output video. Default is True.
            show (bool): Whether to display live feed during detection. Default is True.
            classes (list): List of class IDs to detect. Default is None (detect all classes).
        """
        results = self.model(video_path, save=save, show=show, classes=classes, persist=True)
        return results
