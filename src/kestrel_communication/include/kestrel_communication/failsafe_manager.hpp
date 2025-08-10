#ifndef KESTREL_COMMUNICATION__FAILSAFE_MANAGER_HPP_
#define KESTREL_COMMUNICATION__FAILSAFE_MANAGER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/flight_status.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "kestrel_msgs/srv/set_flight_mode.hpp"
#include <string>
#include <atomic>

namespace kestrel_communication
{

class FailsafeManagerNode : public rclcpp::Node
{
public:
  explicit FailsafeManagerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void flight_status_callback(const kestrel_msgs::msg::FlightStatus::SharedPtr msg);
  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
  void execute_failsafe_action(const std::string& action, const std::string& reason);

  // subscribers for topics that could trigger a failsafe
  rclcpp::Subscription<kestrel_msgs::msg::FlightStatus>::SharedPtr flight_status_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;

  // client to command the drone to change its flight mode (ex.. to RTL or LAND)
  rclcpp::Client<kestrel_msgs::srv::SetFlightMode>::SharedPtr set_flight_mode_client_;

  // failsafe parameters loaded from yaml
  std::string on_low_battery_action_;
  std::string on_gps_loss_action_;
  double low_battery_threshold_;

  // flags to prevent triggering the same failsafe repeatedly
  std::atomic<bool> low_battery_failsafe_triggered_;
  std::atomic<bool> gps_loss_failsafe_triggered_;
};

} // namespace kestrel_communication
#endif // KESTREL_COMMUNICATION__FAILSAFE_MANAGER_HPP_