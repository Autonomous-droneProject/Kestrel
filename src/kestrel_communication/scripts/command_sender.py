import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

# this progam p good for quick repeatable tests
# since it just sends a single pre defined command to the base station node
# you can use it to test the command handling logic without needing to run the full GCS

class CommandSender(Node):
    def __init__(self):
        super().__init__('command_sender')
        self.publisher_ = self.create_publisher(String, 'gcs/command', 10)
        
        # wait a moment for the publisher to connect
        time.sleep(1)

        msg = String()
        msg.data = "SET_MODE:AUTO"
        self.publisher_.publish(msg)
        self.get_logger().info(f'Sent command: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    command_sender = CommandSender()
    # node will publish once and then be destroyed
    command_sender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
