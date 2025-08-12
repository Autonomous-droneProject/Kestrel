#include "kestrel_control/command_validator.hpp"

// Run node
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<kestrel_control::CommandValidator>());
  rclcpp::shutdown();
  return 0;
}
