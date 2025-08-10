import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# command line interface to send commands to the GCS
# im just using this as a debugger lmao
class GcsCli(Node):
    def __init__(self):
        super().__init__('gcs_cli')
        self.publisher_ = self.create_publisher(String, 'gcs/command', 10)
        self.get_logger().info('GCS Command Line Interface started.')
        self.get_logger().info('Enter commands (e.g., "SET_MODE:AUTO", "EMERGENCY_STOP"). Type "exit" to quit.')
        self.run()

    def run(self):
        while rclpy.ok():
            try:
                command = input('> ')
                if command.lower() == 'exit':
                    break
                msg = String()
                msg.data = command
                self.publisher_.publish(msg)
                self.get_logger().info(f'Sent: "{command}"')
            except (KeyboardInterrupt, EOFError):
                break

def main(args=None):
    rclpy.init(args=args)
    gcs_cli = GcsCli()
    gcs_cli.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
