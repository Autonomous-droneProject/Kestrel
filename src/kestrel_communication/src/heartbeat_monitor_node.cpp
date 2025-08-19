#include "kestrel_communication/heartbeat_monitor.hpp"
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_communication::HeartbeatMonitorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}