# Clustering node will subscribe to the tracking node and emit a message called TrackCentroid.msg using ROS2 Jazzy
'''
This file will create the clustering node which will send a message called TrackCentroid.msg to the Camera Node
This node susbcribes to the vision node and publishes to the camera node
'''

import rclpy #ROS Common Library Python
from rclpy.node import Node
from kestrel_msgs.msg import TrackCentroid
from kestrel_msgs.msg import ClusteringInput
from kestrel_msgs.msg import ClusteringInputArray



class ClusteringNode(Node):
    def __init__(self): #Constructor
        super().__init__('clustering_node') #Create a node in the ROS graph
        
        queue_size = 10 # Placeholder queue size
        
        self.subscription = self.create_subscription(ClusteringInputArray, '/kestrel/detections', self.vision_callback, queue_size)
        self.publisher_ = self.create_publisher(TrackCentroid, "/kestrel/centroid", queue_size)
        
    
    def calculate_centroid(self,bbox):
    
        x1, y1= bbox[0]
        x2, y2 = bbox[1]
        
        print("Calculating centroid...")
        print(x1, x2, y1, y2)
        
        centroid = (x1+x2)//2 , (y1+y2)//2

        return centroid
    
    def minimum_bounding_rectangle(msg):
         
        for ClusteringInput in msg.input_array:
            max_x= ClusteringInput.x2 if ClusteringInput.x2 > max_x else max_x
            # max_x= ClusteringInput.x1 if ClusteringInput.x1 > max_x else max_x 

            # min_x= ClusteringInput.x2 if ClusteringInput.x2 < min_x else min_x
            min_x= ClusteringInput.x1 if ClusteringInput.x1 < min_x else min_x #

            max_y= ClusteringInput.y2 if ClusteringInput.y2 > max_y else max_y #
            # max_y= ClusteringInput.y1 if ClusteringInput.y1 > max_y else max_y 

            # min_y= ClusteringInput.y2 if ClusteringInput.y2 < min_y else min_y
            min_y= ClusteringInput.y1 if ClusteringInput.y1 < min_y else min_y #
            
        
        bounding_boxes = ((min_x, min_y), (max_x, max_y))
    
    def publish_centroid(self, msg: ClusteringInputArray):
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        bounding_boxes = self.minimum_bounding_rectangle(msg)
        
        center_x, center_y = self.calculate_centroid(bounding_boxes)
        
        track_centroid = TrackCentroid()
        track_centroid.header = msg.header
        
        track_centroid.center_x = center_x
        track_centroid.center_y = center_y

        track_centroid.x1 = min_x
        track_centroid.x2 = max_x
        track_centroid.y1 = min_y
        track_centroid.y2 = max_y
        
        self.publisher_.publish(track_centroid)
        self.get_logger().info(f'Publishing: {track_centroid}')
            
            
    def vision_callback(self, msg: ClusteringInput):
        self.get_logger().info(f'Received: {msg}')
        
        self.publish_centroid(msg)
        

def main():
    rclpy.init(args=None)
    print("Running the clustering node!")
    clustering_node = ClusteringNode()
    rclpy.spin(clustering_node)
    
    clustering_node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()
        