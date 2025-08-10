#include "kestrel_communication/failsafe_manager.hpp"

namespace kestrel_communication
{

FailsafeManagerNode::FailsafeManagerNode(const rclcpp::NodeOptions & options)
: Node("failsafe_manager_node", options),
  low_battery_failsafe_triggered_(false),
  gps_loss_failsafe_triggered_(false)
{
  initialize();
}

void FailsafeManagerNode::initialize()
{
  // load failsafe action parameters from config
  this->declare_parameter<std::string>("failsafe_actions.on_low_battery", "RTL");
  this->declare_parameter<std::string>("failsafe_actions.on_gps_loss", "LAND");
  this->declare_parameter<double>("failsafe_actions.low_battery_threshold_pct", 20.0);
  on_low_battery_action_ = this->get_parameter("failsafe_actions.on_low_battery").as_string();
  on_gps_loss_action_ = this->get_parameter("failsafe_actions.on_gps_loss").as_string();
  low_battery_threshold_ = this->get_parameter("failsafe_actions.low_battery_threshold_pct").as_double();

  // setup subscribers
  flight_status_sub_ = this->create_subscription<kestrel_msgs::msg::FlightStatus>(
    "drone/flight_status", 10, std::bind(&FailsafeManagerNode::flight_status_callback, this, std::placeholders::_1));
  gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
    "sensors/gps", 10, std::bind(&FailsafeManagerNode::gps_callback, this, std::placeholders::_1));

  // setup service client
  set_flight_mode_client_ = this->create_client<kestrel_msgs::srv::SetFlightMode>("services/set_flight_mode");

  RCLCPP_INFO(this->get_logger(), "FailsafeManagerNode initialized.");
}

void FailsafeManagerNode::flight_status_callback(const kestrel_msgs::msg::FlightStatus::SharedPtr msg)
{
  if (msg->battery_percent < low_battery_threshold_) {
    if (!low_battery_failsafe_triggered_.exchange(true)) {
      execute_failsafe_action(on_low_battery_action_, "low battery");
    }
  }
}

void FailsafeManagerNode::gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
    if (!gps_loss_failsafe_triggered_.exchange(true)) {
      execute_failsafe_action(on_gps_loss_action_, "gps signal lost");
    }
  } else {
    // if gps signal is restored, reset the flag!
    gps_loss_failsafe_triggered_ = false;
  }
}

void FailsafeManagerNode::execute_failsafe_action(const std::string& action, const std::string& reason)
{
  RCLCPP_FATAL(this->get_logger(), "FAILSAFE TRIGGERED: %s. Action: %s", reason.c_str(), action.c_str());

  if (!set_flight_mode_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "set flight mode service not available for failsafe action.");
    return;
  }

  auto request = std::make_shared<kestrel_msgs::srv::SetFlightMode::Request>();
  std::string upper_action = action;
  std::transform(upper_action.begin(), upper_action.end(), upper_action.begin(), ::toupper);

  if (upper_action == "RTL" || upper_action == "LAND") {
    request->flight_mode = kestrel_msgs::msg::FlightStatus::MODE_LANDING; // assuming RTL is handled by LAND mode for now
  } else {
    RCLCPP_ERROR(this->get_logger(), "unknown failsafe action: %s", action.c_str());
    return;
  }

  set_flight_mode_client_->async_send_request(request);
}

} // namespace kestrel_communication
