# Clustering node will subscribe to the tracking node and emit a message called TrackCentroid.msg using ROS2 Jazzy
'''
This file will create the clustering node which will send a message called TrackCentroid.msg to the Camera Node
This node susbcribes to the vision node and publishes to the camera node
'''

import rclpy #ROS Common Library Python
from rclpy.node import Node
from std_msgs.msg import String
#from list_detections import calculate_centroid

#Example TrackCentroid Message
TrackCentroidMessage = """
    Header:
    stamp: ...
    frame_id: ...
    centroid:
        x: <float>  # Image-space X (pixels)
        y: <float>  # Image-space Y (pixels)
    bbox:
        x_min: <float>
        y_min: <float>
        x_max: <float>
        y_max: <float>
    member_ids: <array>
"""

class ClusteringNode(Node):
    def __init__(self): #Constructor
        super().__init__('clustering_node') #Create a node in the ROS graph
        
        queue_size = 10 # Placeholder queue size
        
        self.subscription = self.create_subscription(String, '/kestrel/detections', self.vision_callback, queue_size)
        
        self.publisher_ = self.create_publisher(String, "/kestrel/centroid", queue_size)
    
    def publish_centroid(self, bounding_boxes):
        msg = String()
        #x, y = calculate_centroid(bounding_boxes)
        
        msg.data = f"""
            centroid:
            x: {x}  
            y: {y}  
            bbox:
            x_min: <float>
            y_min: <float>
            x_max: <float>
            y_max: <float>
        """
        
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg}')
            
            
    def vision_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
        
        self.publish_centroid(msg)
        



def main():
    rclpy.init(args=None)
    clustering_node = ClusteringNode()
    rclpy.spin(clustering_node)
    
    clustering_node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()
        