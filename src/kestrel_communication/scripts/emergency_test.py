import rclpy
from rclpy.node import Node
from kestrel_msgs.srv import TriggerEmergencyStop

# this script directly calls the emergency stop service to test the failsafe

class EmergencyTest(Node):
    def __init__(self):
        super().__init__('emergency_test')
        self.cli = self.create_client(TriggerEmergencyStop, 'services/trigger_emergency_stop')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('emergency stop service not available, waiting again...')
        
        self.req = TriggerEmergencyStop.Request()

    def send_request(self):
        self.req.reason = "EMERGENCY TEST SCRIPT"
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    emergency_test_cli = EmergencyTest()
    response = emergency_test_cli.send_request()
    
    if response:
        emergency_test_cli.get_logger().info(f'Emergency stop acknowledged: {response.acknowledged}')
    else:
        emergency_test_cli.get_logger().error('Exception while calling service')

    emergency_test_cli.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()