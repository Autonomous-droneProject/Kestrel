#ifndef KESTREL_COMMUNICATION__TELEMETRY_MANAGER_HPP_
#define KESTREL_COMMUNICATION__TELEMETRY_MANAGER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/flight_status.hpp"
#include "kestrel_msgs/msg/system_health.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include <mutex>
#include <fstream>

namespace kestrel_communication
{

class TelemetryManagerNode : public rclcpp::Node
{
public:
  explicit TelemetryManagerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void battery_callback(const sensor_msgs::msg::BatteryState::SharedPtr msg);
  void mavros_state_callback(const mavros_msgs::msg::State::SharedPtr msg);
  void publish_telemetry();
  float get_cpu_temperature();
  float get_memory_usage();
  float get_cpu_usage();

  // subscribers for individual data points
  rclcpp::Subscription<sensor_msgs::msg::BatteryState>::SharedPtr battery_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr mavros_state_sub_;

  // publishers for aggregated telemetry messages
  rclcpp::Publisher<kestrel_msgs::msg::FlightStatus>::SharedPtr flight_status_pub_;
  rclcpp::Publisher<kestrel_msgs::msg::SystemHealth>::SharedPtr system_health_pub_;

  // timer to control the publishing rate
  rclcpp::TimerBase::SharedPtr timer_;

  // data storage with mutex for thread safety
  std::mutex data_mutex_;
  sensor_msgs::msg::BatteryState latest_battery_state_;
  mavros_msgs::msg::State latest_mavros_state_;
  bool has_battery_data_ = false;
  bool has_mavros_state_ = false;

  // time
  long long prev_total_time_;
  long long prev_idle_time_;
};

} // namespace kestrel_communication
#endif // KESTREL_COMMUNICATION__TELEMETRY_MANAGER_HPP_