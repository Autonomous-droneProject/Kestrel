#include "kestrel_communication/base_station_node.hpp"
#include <vector>
#include <sstream>

namespace kestrel_communication
{

BaseStationNode::BaseStationNode(const rclcpp::NodeOptions & options)
: Node("base_station_node", options)
{
  initialize();
}

void BaseStationNode::initialize()
{
  // setup subscribers
  gcs_command_sub_ = this->create_subscription<std_msgs::msg::String>(
    "gcs/command", 10, std::bind(&BaseStationNode::command_callback, this, std::placeholders::_1));
  flight_status_sub_ = this->create_subscription<kestrel_msgs::msg::FlightStatus>(
    "drone/flight_status", 10, std::bind(&BaseStationNode::flight_status_callback, this, std::placeholders::_1));
  system_health_sub_ = this->create_subscription<kestrel_msgs::msg::SystemHealth>(
    "drone/system_health", 10, std::bind(&BaseStationNode::system_health_callback, this, std::placeholders::_1));

  // setup publishers
  gcs_flight_status_pub_ = this->create_publisher<kestrel_msgs::msg::FlightStatus>("gcs/flight_status", 10);
  gcs_system_health_pub_ = this->create_publisher<kestrel_msgs::msg::SystemHealth>("gcs/system_health", 10);

  // setup service clients
  set_flight_mode_client_ = this->create_client<kestrel_msgs::srv::SetFlightMode>("services/set_flight_mode");
  trigger_emergency_stop_client_ = this->create_client<kestrel_msgs::srv::TriggerEmergencyStop>("services/trigger_emergency_stop");

  RCLCPP_INFO(this->get_logger(), "BaseStationNode initialized");
}

// this callback just republishes the drones flight status for the gcs
void BaseStationNode::flight_status_callback(const kestrel_msgs::msg::FlightStatus::SharedPtr msg)
{
  gcs_flight_status_pub_->publish(*msg);
}

// this callback just republishes the drones system health for the gcs
void BaseStationNode::system_health_callback(const kestrel_msgs::msg::SystemHealth::SharedPtr msg)
{
  gcs_system_health_pub_->publish(*msg);
}

// this callback handles incoming commands from the gcs
void BaseStationNode::command_callback(const std_msgs::msg::String::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "received command: '%s'", msg->data.c_str());

  std::string command = msg->data;
  std::transform(command.begin(), command.end(), command.begin(), ::toupper);

  if (command.rfind("SET_MODE:", 0) == 0) {
    // example command: "SET_MODE:AUTO"
    std::string mode = command.substr(9);
    handle_set_mode_command(mode);
  }
  else if (command == "EMERGENCY_STOP") {
    if (!trigger_emergency_stop_client_->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_ERROR(this->get_logger(), "emergency stop service not available.");
      return;
    }
    auto request = std::make_shared<kestrel_msgs::srv::TriggerEmergencyStop::Request>();
    request->reason = "command from gcs";
    trigger_emergency_stop_client_->async_send_request(request);
  }
  else {
    RCLCPP_WARN(this->get_logger(), "unknown command received: '%s'", msg->data.c_str());
  }
}

void BaseStationNode::handle_set_mode_command(const std::string& mode_str)
{
  if (!set_flight_mode_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "set flight mode service not available.");
    return;
  }

  auto request = std::make_shared<kestrel_msgs::srv::SetFlightMode::Request>();

  // this is really ugly i know
  // but i need to convert the string to the enum value
  // i dont think theres any other way to do this in ROS2
  
  if (mode_str == "MANUAL") {
    request->flight_mode = kestrel_msgs::msg::FlightStatus::MODE_MANUAL;
  } else if (mode_str == "AUTO") {
    request->flight_mode = kestrel_msgs::msg::FlightStatus::MODE_AUTO;
  } else if (mode_str == "LAND") {
    request->flight_mode = kestrel_msgs::msg::FlightStatus::MODE_LANDING;
  } else {
    RCLCPP_WARN(this->get_logger(), "unknown flight mode: '%s'", mode_str.c_str());
    return;
  }

  set_flight_mode_client_->async_send_request(request);
}

} // namespace kestrel_communication

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_communication::BaseStationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
