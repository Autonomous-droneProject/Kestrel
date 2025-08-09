#ifndef KESTREL_COMMUNICATION__BASE_STATION_NODE_HPP_
#define KESTREL_COMMUNICATION__BASE_STATION_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/srv/set_flight_mode.hpp"
#include "kestrel_msgs/srv/trigger_emergency_stop.hpp"
#include "kestrel_msgs/msg/flight_status.hpp"
#include "kestrel_msgs/msg/system_health.hpp"
#include "std_msgs/msg/string.hpp"

 // NOTE, THIS IS NOT MY AREA OF EXPERTISE
 // ILL TRY MY BEST BASED ON TUTORIALS IVE FOLLOWED
namespace kestrel_communication
{

class BaseStationNode : public rclcpp::Node
{
public:
  explicit BaseStationNode(const rclcpp::NodeOptions & options);

private:
  void initialize();
  void command_callback(const std_msgs::msg::String::SharedPtr msg);
  void flight_status_callback(const kestrel_msgs::msg::FlightStatus::SharedPtr msg);
  void system_health_callback(const kestrel_msgs::msg::SystemHealth::SharedPtr msg);
  void handle_set_mode_command(const std::string& mode_str);

  // subscribers for commands from gcs and internal telemetry
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gcs_command_sub_;
  rclcpp::Subscription<kestrel_msgs::msg::FlightStatus>::SharedPtr flight_status_sub_;
  rclcpp::Subscription<kestrel_msgs::msg::SystemHealth>::SharedPtr system_health_sub_;

  // publishers to send telemetry to the gcs
  rclcpp::Publisher<kestrel_msgs::msg::FlightStatus>::SharedPtr gcs_flight_status_pub_;
  rclcpp::Publisher<kestrel_msgs::msg::SystemHealth>::SharedPtr gcs_system_health_pub_;

  // service clients to send commands to the drone's internal systems
  rclcpp::Client<kestrel_msgs::srv::SetFlightMode>::SharedPtr set_flight_mode_client_;
  rclcpp::Client<kestrel_msgs::srv::TriggerEmergencyStop>::SharedPtr trigger_emergency_stop_client_;
};

} // namespace kestrel_communication
#endif // KESTREL_COMMUNICATION__BASE_STATION_NODE_HPP_