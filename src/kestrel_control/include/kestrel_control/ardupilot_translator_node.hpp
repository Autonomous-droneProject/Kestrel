#ifndef KESTREL_CONTROL__ARDUPILOT_TRANSLATOR_NODE_HPP_
#define KESTREL_CONTROL__ARDUPILOT_TRANSLATOR_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mavros_msgs/msg/position_target.hpp"
#include "kestrel_msgs/srv/trigger_emergency_stop.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "kestrel_msgs/srv/set_flight_mode.hpp"

namespace kestrel_control
{

class ArduPilotTranslatorNode : public rclcpp::Node
{
public:
  explicit ArduPilotTranslatorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void waypoint_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void handle_emergency_stop(
    const std::shared_ptr<kestrel_msgs::srv::TriggerEmergencyStop::Request> request,
    std::shared_ptr<kestrel_msgs::srv::TriggerEmergencyStop::Response> response);
  void handle_set_flight_mode(
    const std::shared_ptr<kestrel_msgs::srv::SetFlightMode::Request> request,
    std::shared_ptr<kestrel_msgs::srv::SetFlightMode::Response> response);

  // subscribes to high-level waypoints from the path planner
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr waypoint_sub_;

  // publishes low-level commands that mavros understands
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr target_pub_;

  rclcpp::Service<kestrel_msgs::srv::TriggerEmergencyStop>::SharedPtr emergency_stop_service_;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
  rclcpp::Service<kestrel_msgs::srv::SetFlightMode>::SharedPtr set_flight_mode_service_;
};

} // namespace kestrel_control
#endif // KESTREL_CONTROL__ARDUPILOT_TRANSLATOR_NODE_HPP_