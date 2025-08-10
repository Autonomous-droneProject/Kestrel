#ifndef KESTREL_COMMUNICATION__HEARTBEAT_MONITOR_HPP_
#define KESTREL_COMMUNICATION__HEARTBEAT_MONITOR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp" // using header for timestamped heartbeat... makes it easier to track
#include "kestrel_msgs/srv/trigger_emergency_stop.hpp"
#include <atomic>

namespace kestrel_communication
{

class HeartbeatMonitorNode : public rclcpp::Node
{
public:
  explicit HeartbeatMonitorNode(const rclcpp::NodeOptions & options);

private:
  void initialize();
  void heartbeat_callback(const std_msgs::msg::Header::SharedPtr msg);
  void check_connection_callback();
  void trigger_failsafe(const std::string& reason);

  // subscribes to the heartbeat topic from the gcs
  rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr heartbeat_sub_;

  // client to call the emergency stop service
  rclcpp::Client<kestrel_msgs::srv::TriggerEmergencyStop>::SharedPtr estop_client_;

  // timer to periodically check the connection status
  rclcpp::TimerBase::SharedPtr timer_;

  // parameters
  double heartbeat_timeout_;

  // state
  rclcpp::Time last_heartbeat_time_;
  std::atomic<bool> connection_lost_;
};

} // namespace kestrel_communication
#endif // KESTREL_COMMUNICATION__HEARTBEAT_MONITOR_HPP_
