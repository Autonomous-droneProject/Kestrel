#ifndef KESTREL_CONTROL__ARDUPILOT_TRANSLATOR_NODE_HPP_
#define KESTREL_CONTROL__ARDUPILOT_TRANSLATOR_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mavros_msgs/msg/position_target.hpp"

namespace kestrel_control
{

class ArduPilotTranslatorNode : public rclcpp::Node
{
public:
  explicit ArduPilotTranslatorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void waypoint_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  // subscribes to high-level waypoints from the path planner
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr waypoint_sub_;

  // publishes low-level commands that mavros understands
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr target_pub_;
};

} // namespace kestrel_control
#endif // KESTREL_CONTROL__ARDUPILOT_TRANSLATOR_NODE_HPP_