#include "kestrel_communication/failsafe_manager.hpp"
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_communication::FailsafeManagerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}