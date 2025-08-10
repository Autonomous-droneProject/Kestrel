import rclpy
from rclpy.node import Node
from kestrel_msgs.msg import FlightStatus, SystemHealth
import os
import time

# ok so this SEEMS similar to telemetry_logger.py
# but this prints the LATEST data to the console
# while telemetry_logger.py is more like a historical record
# this is for REAL time monitoring! 

class TelemetryViewer(Node):
    def __init__(self):
        super().__init__('telemetry_viewer')
        self.flight_status_sub = self.create_subscription(
            FlightStatus, 'drone/flight_status', self.flight_status_callback, 10)
        self.system_health_sub = self.create_subscription(
            SystemHealth, 'drone/system_health', self.system_health_callback, 10)
        
        self.latest_flight_status = None
        self.latest_system_health = None

        self.timer = self.create_timer(0.5, self.update_display) # refresh twice a second
        self.get_logger().info('Telemetry Viewer started.')

    def flight_status_callback(self, msg):
        self.latest_flight_status = msg

    def system_health_callback(self, msg):
        self.latest_system_health = msg

    def update_display(self):
        # clear the console screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("Kestrel Telemetry Viewer")
        
        if self.latest_flight_status:
            fs = self.latest_flight_status
            print(f"Flight Status | Armed: {fs.armed} | Mode: {fs.flight_mode} | Battery: {fs.battery_percent:.1f}%")
        else:
            print("Flight Status | Waiting for data...")

        if self.latest_system_health:
            sh = self.latest_system_health
            print(f"System Health | Temp: {sh.temperature_celsius:.1f}C | CPU: {sh.cpu_usage_percent:.1f}% | Mem: {sh.memory_usage_percent:.1f}%")
        else:
            print("System Health | Waiting for data...")

def main(args=None):
    rclpy.init(args=args)
    telemetry_viewer = TelemetryViewer()
    try:
        rclpy.spin(telemetry_viewer)
    except KeyboardInterrupt:
        pass
    finally:
        telemetry_viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()