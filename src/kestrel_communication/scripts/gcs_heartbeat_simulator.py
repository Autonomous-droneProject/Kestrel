#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

class GCSHeartbeatSimulator(Node):
    def __init__(self):
        super().__init__('gcs_heartbeat_simulator')
        self.publisher = self.create_publisher(Header, 'gcs/heartbeat', 10)
        self.timer = self.create_timer(1.0, self.publish_heartbeat)  # 1Hz heartbeat
        
    def publish_heartbeat(self):
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = "gcs"
        self.publisher.publish(msg)
        
def main():
    rclpy.init()
    node = GCSHeartbeatSimulator()
    rclpy.spin(node)
    
if __name__ == '__main__':
    main()