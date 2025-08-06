# ROS and system libraries
import rclpy
from rclpy.node import Node
import sys
from pathlib import Path

# ROS messages
from sensor_msgs.msg import Image
from kestrel_msgs.msg import DetectionWithEmbedding, DetectionWithEmbeddingArray
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, BoundingBox2D

# Computer vision and deep learning libraries
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from ultralytics import YOLO


from model import CNNdeepSORT
from . import processing


class VisionNode(Node):
    """
    This node subscribes to an image topic, runs YOLO and cnn,
    and publishes the detections with their feature embeddings.
    """
    def __init__(self):
        super().__init__('vision_node')

        # initialize models
        self.yolo_model = YOLO('yolo11m.pt')
        self.cnn_model = CNNdeepSORT(embedding_dim=128)
        self.get_logger().info("Loaded YOLO and CNNdeepSORT models successfully.")

        self.bridge = CvBridge()
        # threshold to stop certain stuff to go into cnn
        self.conf_threshold = self.declare_parameter('conf_threshold', 0.5).get_parameter_value().double_value

        # create publishers and subscription
        self.pub_det = self.create_publisher(DetectionWithEmbeddingArray, '/kestrel/detections', 10)
        self.pub_dbg = self.create_publisher(Image, '/kestrel/debug/frame', 10)
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

    def image_callback(self, msg: Image):
        """
        Callback to process every frame
        """
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        yolo_results = self.yolo_model(frame, verbose=False)[0]

        det_array_msg = DetectionWithEmbeddingArray()
        det_array_msg.header = msg.header

        for box in yolo_results.boxes:
            conf = float(box.conf[0])
            class_name = self.yolo_model.names[int(box.cls[0])]

            # checks if person and confidence is higher than threshold
            if class_name == 'person' and conf >= self.conf_threshold:
                
                appearance_vector = processing.extract_person_embedding(
                    frame, box, self.cnn_model
                )

                if appearance_vector is None:
                    continue

                # Populate the custom ROS message
                custom_detection = DetectionWithEmbedding()
                custom_detection.detection.header = msg.header
                
                # standard hypothesis msg from ros with info on detection 
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = conf
                custom_detection.detection.results.append(hypothesis)

                x1, y1, x2, y2 = map(int, box.xyxy[0]) # get corners of bbox
                custom_detection.detection.bbox = BoundingBox2D(
                    center=rclpy.geometry.Pose2D(x=float((x1+x2)/2), y=float((y1+y2)/2)),
                    size_x=float(x2-x1),
                    size_y=float(y2-y1)
                )

                # add this detection to rest of all detections (the array)
                custom_detection.embedding = appearance_vector.tolist()
                det_array_msg.detections.append(custom_detection)

                # draws bbox on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish the results
        self.pub_det.publish(det_array_msg)
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))


def main(args=None):
    """Standard entry point for the ROS node."""
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()