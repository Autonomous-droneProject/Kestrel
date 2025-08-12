#include "kestrel_communication/emergency_handler.hpp"
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_communication::EmergencyHandlerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}