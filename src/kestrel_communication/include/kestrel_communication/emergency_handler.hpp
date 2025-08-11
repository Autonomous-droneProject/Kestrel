#ifndef KESTREL_COMMUNICATION__EMERGENCY_HANDLER_HPP_
#define KESTREL_COMMUNICATION__EMERGENCY_HANDLER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/proximity_alert.hpp"
#include "kestrel_msgs/msg/flight_status.hpp"
#include "kestrel_msgs/srv/trigger_emergency_stop.hpp"
#include <atomic>

namespace kestrel_communication
{

class EmergencyHandlerNode : public rclcpp::Node
{
public:
  explicit EmergencyHandlerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void proximity_callback(const kestrel_msgs::msg::ProximityAlert::SharedPtr msg);
  void flight_status_callback(const kestrel_msgs::msg::FlightStatus::SharedPtr msg);
  void trigger_emergency_stop(const std::string& reason);

  // subscribers for topics that COULD trigger an emergency
  rclcpp::Subscription<kestrel_msgs::msg::ProximityAlert>::SharedPtr proximity_sub_;
  rclcpp::Subscription<kestrel_msgs::msg::FlightStatus>::SharedPtr flight_status_sub_;

  // client to call the emergency stop service
  rclcpp::Client<kestrel_msgs::srv::TriggerEmergencyStop>::SharedPtr estop_client_;

  // emergency trigger parameters
  double critical_proximity_distance_;
  double critical_battery_level_;

  // flag to prevent sending multiple emergency stop commands
  std::atomic<bool> emergency_triggered_;
};

} // namespace kestrel_communication
#endif // KESTREL_COMMUNICATION__EMERGENCY_HANDLER_HPP_