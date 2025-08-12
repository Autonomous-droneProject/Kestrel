#include "kestrel_communication/heartbeat_monitor.hpp"

namespace kestrel_communication
{

HeartbeatMonitorNode::HeartbeatMonitorNode(const rclcpp::NodeOptions & options)
: Node("heartbeat_monitor_node", options), connection_lost_(false)
{
  initialize();
}

void HeartbeatMonitorNode::initialize()
{
  // load parameters
  this->declare_parameter<double>("comms.heartbeat_timeout_s", 5.0);
  heartbeat_timeout_ = this->get_parameter("comms.heartbeat_timeout_s").as_double();

  // setup subscriber
  heartbeat_sub_ = this->create_subscription<std_msgs::msg::Header>(
    "gcs/heartbeat", 10, std::bind(&HeartbeatMonitorNode::heartbeat_callback, this, std::placeholders::_1));

  // setup service client
  estop_client_ = this->create_client<kestrel_msgs::srv::TriggerEmergencyStop>("services/trigger_emergency_stop");

  // setup timer to check connection status
  timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&HeartbeatMonitorNode::check_connection_callback, this));

  // initialize last heartbeat time to now to prevent immediate timeout
  last_heartbeat_time_ = this->get_clock()->now();

  RCLCPP_INFO(this->get_logger(), "HeartbeatMonitorNode initialized.");
}

void HeartbeatMonitorNode::heartbeat_callback(const std_msgs::msg::Header::SharedPtr msg)
{
  // update the time of the last received heartbeat
  last_heartbeat_time_ = msg->stamp;

  // if connection was previously lost, log that it has been restored!!!
  if (connection_lost_.exchange(false)) {
    RCLCPP_INFO(this->get_logger(), "gcs connection restored.");
  }
}

void HeartbeatMonitorNode::check_connection_callback()
{
  // if a failsafe has already been triggered, do nothing!!!
  if (connection_lost_) {
    return;
  }

  rclcpp::Duration time_since_heartbeat = this->get_clock()->now() - last_heartbeat_time_;

  if (time_since_heartbeat.seconds() > heartbeat_timeout_) {
    trigger_failsafe("gcs connection lost (heartbeat timeout)");
  }
}

void HeartbeatMonitorNode::trigger_failsafe(const std::string& reason)
{
  // setting the flag to true to prevent this from being called multiple times
  connection_lost_ = true;
  RCLCPP_FATAL(this->get_logger(), "FAILSAFE TRIGGERED: %s", reason.c_str());

  if (!estop_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "emergency stop service not available for failsafe.");
    return;
  }

  auto request = std::make_shared<kestrel_msgs::srv::TriggerEmergencyStop::Request>();
  request->reason = reason;
  estop_client_->async_send_request(request);
}

} // namespace kestrel_communication

