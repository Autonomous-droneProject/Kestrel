# ROS and system libraries
import rclpy
from rclpy.node import Node
import sys
from pathlib import Path

# ROS messages
from sensor_msgs.msg import Image
from kestrel_msgs.msg import Detection, DetectionArray

# Computer vision and deep learning libraries
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from ultralytics import YOLO


from .model import CNNdeepSORT
from . import processing


class VisionNode(Node):
    """
    This node subscribes to an image topic, runs YOLO and CNN,
    and publishes the detections with their feature embeddings.
    """
    def __init__(self):
        super().__init__('vision_node')

       # initialize models
        yolo_path = Path(__file__).parent/'yolov8n.pt'
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to('cpu') #TODO: why CPU?

        # ---- CNN init  ----
        self.cnn_model = CNNdeepSORT(embedding_dim=128)

        checkpoint_path = Path(__file__).parent / 'best_model_checkpoint.pth'
        if checkpoint_path.is_file():
            try:
                # torch.load with weights_only when available (2.1+), else fallback
                try:
                    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                except TypeError:
                    ckpt = torch.load(checkpoint_path, map_location='cpu')

                # Find the actual state_dict inside common wrappers
                state = None
                if isinstance(ckpt, dict):
                    for k in ('model_state_dict', 'state_dict', 'net', 'model'):
                        if k in ckpt and isinstance(ckpt[k], dict):
                            state = ckpt[k]
                            break
                    if state is None:
                        # maybe it's already a raw state_dict
                        state = {k: v for k, v in ckpt.items() if hasattr(v, 'dtype')}
                if state is None:
                    raise RuntimeError("Unrecognized checkpoint format")

                # Clean keys: strip 'module.' and drop classifier / batch-tracking
                from collections import OrderedDict
                cleaned = OrderedDict()
                for k, v in state.items():
                    if k.startswith('module.'):
                        k = k[len('module.'):]
                    if k.startswith('classifier.') or k.endswith('num_batches_tracked'):
                        continue
                    cleaned[k] = v

                # Filter to keys that exist in the current model
                model_sd = self.cnn_model.state_dict()
                intersect = {k: v for k, v in cleaned.items() if k in model_sd and v.shape == model_sd[k].shape}

                # Informative logs
                dropped_unexpected = sorted(set(cleaned.keys()) - set(intersect.keys()))
                missing_in_ckpt = sorted(set(model_sd.keys()) - set(intersect.keys()))
                if dropped_unexpected:
                    self.get_logger().warn(f"Dropping {len(dropped_unexpected)} checkpoint keys not in the current model (e.g., {dropped_unexpected[:3]})")
                if missing_in_ckpt:
                    self.get_logger().warn(f"{len(missing_in_ckpt)} model keys not found/mismatched in checkpoint (e.g., {missing_in_ckpt[:3]})")

                # Load with strict=False so non-intersecting layers stay at init
                self.cnn_model.load_state_dict(intersect, strict=False)
                self.get_logger().info("Loaded CNN weights (filtered to matching layers).")
            except Exception as e:
                self.get_logger().error(f"Failed to load CNN checkpoint: {e}. Using randomly initialized weights.")
        else:
            self.get_logger().warn(f"Could not find CNN checkpoint at {checkpoint_path}. Using untrained model.")

        self.cnn_model.eval()
        self.cnn_model.to('cpu')   # or to('cuda') later via a device param
        # -----------------------------------------------------------------------



        self.bridge = CvBridge()
        # threshold to stop certain stuff to go into cnn
        self.conf_threshold = self.declare_parameter('conf_threshold', 0.5).get_parameter_value().double_value

        # create publishers and subscription
        self.pub_det = self.create_publisher(DetectionArray, '/kestrel/detections', 10)
        self.pub_dbg = self.create_publisher(Image, '/kestrel/debug/frame', 10)
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

    def image_callback(self, msg: Image):
        """
        Callback to process every frame
        """
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        yolo_results = self.yolo_model(frame, verbose=False)[0]

        det_array_msg = DetectionArray()
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
                detection_msg = Detection()
                detection_msg.header = msg.header
                
                # populate msg with info on detection 
                detection_msg.class_name = class_name
                detection_msg.conf = conf
                
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()] # get corners of bbox
                detection_msg.x1 = float(x1)
                detection_msg.y1 = float(y1)
                detection_msg.x2 = float(x2)
                detection_msg.y2 = float(y2)
                
                detection_msg.w = float(x2 - x1)
                detection_msg.h = float(y2 - y1)

                detection_msg.center_x = float((x1 + x2) / 2.0)
                detection_msg.center_y = float((y1 + y2) / 2.0)
                
                emb = appearance_vector.reshape(-1)

                if emb.shape[0] != 128:
                    self.get_logger().warn(f"Bad embedding length: {emb.shape[0]}")
                    continue

                # add this detection to rest of all detections (the array)
                detection_msg.embedding = [float(v) for v in emb.tolist()]
                det_array_msg.detections.append(detection_msg)

                # draws bbox on the frame
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1i, y1i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        node.get_logger().info("Shutting down on Ctrl-C")
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():   # not already shut down
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()