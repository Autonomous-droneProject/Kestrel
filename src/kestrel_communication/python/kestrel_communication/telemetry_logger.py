import rclpy
from rclpy.node import Node
from kestrel_msgs.msg import FlightStatus, SystemHealth

class TelemetryLoggerNode(Node):
    def __init__(self):
        super().__init__('telemetry_logger_node')
        self.create_subscription(FlightStatus, 'drone/flight_status', self.flight_status_callback, 10)
        self.create_subscription(SystemHealth, 'drone/system_health', self.system_health_callback, 10)
        self.get_logger().info('TelemetryLoggerNode has been started.')

    def flight_status_callback(self, msg):
        self.get_logger().info(
            f'Flight Status | Armed: {msg.armed}, Mode: {msg.flight_mode}, Battery: {msg.battery_percent:.1f}%'
        )

    def system_health_callback(self, msg):
        self.get_logger().info(
            f'System Health | Status: {msg.status}, CPU Temp: {msg.temperature_celsius:.1f}C, '
            f'CPU Usage: {msg.cpu_usage_percent:.1f}%, Mem Usage: {msg.memory_usage_percent:.1f}%'
        )

def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()