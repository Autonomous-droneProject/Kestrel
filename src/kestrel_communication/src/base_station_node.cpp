#include "kestrel_communication/base_station.hpp"
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_communication::BaseStationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}