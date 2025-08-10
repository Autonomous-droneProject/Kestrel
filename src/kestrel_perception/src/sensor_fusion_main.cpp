#include "kestrel_perception/sensor_fusion_node.hpp" // Or wherever your node's header is
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  // Instantiate the Node and spin it
  rclcpp::spin(std::make_shared<kestrel_perception::SensorFusionNode>());
  rclcpp::shutdown();
  return 0;
}