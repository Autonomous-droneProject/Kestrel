#include "kestrel_control/frame_transformer.hpp"
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_control::FrameTransformer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}