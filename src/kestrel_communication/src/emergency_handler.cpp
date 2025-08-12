#include "kestrel_communication/emergency_handler.hpp"

namespace kestrel_communication
{

EmergencyHandlerNode::EmergencyHandlerNode(const rclcpp::NodeOptions & options)
: Node("emergency_handler_node", options), emergency_triggered_(false)
{
  initialize();
}

void EmergencyHandlerNode::initialize()
{
  // load emergency trigger parameters from a config file
  this->declare_parameter<double>("emergency_triggers.critical_proximity_m", 0.5);
  this->declare_parameter<double>("emergency_triggers.critical_battery_pct", 10.0);
  critical_proximity_distance_ = this->get_parameter("emergency_triggers.critical_proximity_m").as_double();
  critical_battery_level_ = this->get_parameter("emergency_triggers.critical_battery_pct").as_double();

  // setup subscribers
  proximity_sub_ = this->create_subscription<kestrel_msgs::msg::ProximityAlert>(
    "perception/proximity_alert", 10, std::bind(&EmergencyHandlerNode::proximity_callback, this, std::placeholders::_1));
  flight_status_sub_ = this->create_subscription<kestrel_msgs::msg::FlightStatus>(
    "drone/flight_status", 10, std::bind(&EmergencyHandlerNode::flight_status_callback, this, std::placeholders::_1));

  // setup service client
  estop_client_ = this->create_client<kestrel_msgs::srv::TriggerEmergencyStop>("services/trigger_emergency_stop");

  RCLCPP_INFO(this->get_logger(), "EmergencyHandlerNode initialized.");
}

void EmergencyHandlerNode::proximity_callback(const kestrel_msgs::msg::ProximityAlert::SharedPtr msg)
{
  if (msg->distance < critical_proximity_distance_) {
    trigger_emergency_stop("critical proximity alert");
  }
}

void EmergencyHandlerNode::flight_status_callback(const kestrel_msgs::msg::FlightStatus::SharedPtr msg)
{
  if (msg->battery_percent < critical_battery_level_) {
    trigger_emergency_stop("critical battery level");
  }
}

void EmergencyHandlerNode::trigger_emergency_stop(const std::string& reason)
{
  // check if an emergency has already been triggered to AVOID spamming the service
  if (emergency_triggered_.exchange(true)) {
    return; // an emergency is already in progress!!!!!
  }

  RCLCPP_FATAL(this->get_logger(), "EMERGENCY TRIGGERED: %s", reason.c_str());

  if (!estop_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "emergency stop service not available.");
    emergency_triggered_ = false; // this is a little jank but this is a reset flag if the service is unavailable
    return;
  }

  auto request = std::make_shared<kestrel_msgs::srv::TriggerEmergencyStop::Request>();
  request->reason = reason;
  estop_client_->async_send_request(request);
}

} // namespace kestrel_communication