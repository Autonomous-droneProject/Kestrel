# Clustering node will subscribe to the tracking node and emit a message called TrackCenter.msg using ROS2 Jazzy
'''
This file will create the clustering node which will send a message called TrackCenter.msg to the next node
This node susbcribes to the input node and publishes to the next node
'''
import rclpy #ROS Common Library Python
from rclpy.node import Node
from kestrel_msgs.msg import TrackCenter
from kestrel_msgs.msg import TrackArray


#GLOBAL VARIABLES
#Topics
subscribed_topic = "/kestrel/tracks" #David may need to change this once the input node is implemented
published_topic = "/kestrel/track_center" #The next node in the ROS graph needs to subscribe to this one


class ClusteringNode(Node):
    def __init__(self): #Constructor
        super().__init__('clustering_node') #Create a node in the ROS graph
        
        queue_size = 10 # Default queue size

        self.subscription = self.create_subscription(TrackArray, subscribed_topic, self.vision_callback, queue_size)
        self.publisher_ = self.create_publisher(TrackCenter, published_topic, queue_size)
        
    
    def _calculate_center(self, x1, y1, x2, y2):
        return (x1+x2)/2 , (y1+y2)/2
        
    
    def _minimum_bounding_rectangle(self, input_message):
        #Image coordinate system involves top left of the frame being the origin
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for track in input_message.tracks:

            #Instead of comparing 8 points, we realized that we can half that number with the following logic:
            #Take max_x for example, x2 being the bottom-right corner of any bbox, meaning it's x value will always be larger than x1
            #So we need only compare max_x with x2 and there is no need to compare max_x with x1. This logic applies both ways for all points
            max_x = track.x2 if track.x2 > max_x else max_x
            min_x = track.x1 if track.x1 < min_x else min_x
            max_y = track.y2 if track.y2 > max_y else max_y
            min_y = track.y1 if track.y1 < min_y else min_y

        
        if min_x == float('inf'):
            return None
        
        return (min_x, min_y, max_x, max_y)
    

    def _publish_center(self, input_message: TrackArray):
        rectangle = self._minimum_bounding_rectangle(input_message)
        
        if rectangle == None:
            self.get_logger().debug("No tracks in message. Skipping publish.")
            return
        
        x1, y1, x2, y2 = rectangle
        
        #Calculate the center and the extents of the minimum bounding rectangle
        center_x, center_y = self._calculate_center(x1, y1, x2, y2)
        
        #Initialize the output message
        output_message = TrackCenter()

        #Default parameters that come from the TrackArray message
        output_message.header = input_message.header

        #Newly calculated parameters, unique to the clustering node:
        #Top left corner and bottom right corner
        output_message.x1 = x1
        output_message.y1 = y1
        output_message.x2 = x2
        output_message.y2 = y2

        #Center of the minimum bounding rectangle
        output_message.center_x = center_x
        output_message.center_y = center_y

        self.publisher_.publish(output_message)
        self.get_logger().info(f'Publishing: {output_message}')
            
            
    def vision_callback(self, input_message: TrackArray):
        self.get_logger().info(f'Received: {input_message}')
        self._publish_center(input_message) #Call the function that will then create the TrackCenter message and publish it
        

def main():
    rclpy.init(args=None)
    print("Running the clustering node!")
    clustering_node = ClusteringNode()
    rclpy.spin(clustering_node)
    
    clustering_node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()