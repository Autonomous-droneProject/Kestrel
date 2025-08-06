# kestrel_vision/vision_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray        # standard ROS message :contentReference[oaicite:2]{index=2}
from kestrel_msgs.msg import EmbeddedDetection2D, EmbeddedDetection2DArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from model import CNNdeepSORT
import cv2

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.pub_det = self.create_publisher(Detection2DArray,
                                             '/kestrel/detections', 10)
        self.pub_dbg = self.create_publisher(Image,
                                             '/kestrel/debug/frame', 1)
        self.timer = self.create_timer(1/30.0, self.infer)   # 30 Hz
        self.bridge = CvBridge()
        model_path = self.declare_parameter('model', 'yolov8n.pt').value
        self.model = YOLO(model_path)

        # if reading from webcam; later subscribe to '/camera/image_raw'
        #self.cap = cv2.VideoCapture(0)

    def infer(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        for result in self.model(frame):
            det_msg = Detection2DArray()
            det_msg.header.stamp = self.get_clock().now().to_msg()
            # loop over result.boxes.xyxy etc. → fill det_msg.detections
            # convert bbox xyxy → center, width, height
        self.pub_det.publish(det_msg)
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()