#ifndef KESTREL_CONTROL__COMMAND_VALIDATOR_HPP_
#define KESTREL_CONTROL__COMMAND_VALIDATOR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace kestrel_control
{

class CommandValidator : public rclcpp::Node
{
public:
  explicit CommandValidator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void command_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  bool is_safe(const geometry_msgs::msg::Twist & command);

  // ros interfaces
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr unsafe_cmd_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr safe_cmd_pub_;

  // safety parameters
  double max_linear_speed_;
  double max_angular_speed_;
  double geofence_min_x_, geofence_max_x_;
  double geofence_min_y_, geofence_max_y_;
  double geofence_min_z_, geofence_max_z_;

  // current state
  geometry_msgs::msg::PoseStamped current_pose_;
  bool has_pose_ = false;
};

} // namespace kestrel_control
#endif // KESTREL_CONTROL__COMMAND_VALIDATOR_HPP_