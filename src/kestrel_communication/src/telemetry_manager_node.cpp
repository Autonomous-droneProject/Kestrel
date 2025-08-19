#include "kestrel_communication/telemetry_manager.hpp"
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_communication::TelemetryManagerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}