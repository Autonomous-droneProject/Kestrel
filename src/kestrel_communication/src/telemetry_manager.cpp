#include "kestrel_communication/telemetry_manager.hpp"
#include <string>
#include <vector>
#include <sstream>

namespace kestrel_communication
{

TelemetryManagerNode::TelemetryManagerNode(const rclcpp::NodeOptions & options)
: Node("telemetry_manager_node", options)
{
  initialize();
}

void TelemetryManagerNode::initialize()
{
  // setup subscribers
  battery_sub_ = this->create_subscription<sensor_msgs::msg::BatteryState>(
    "mavros/battery", 10, std::bind(&TelemetryManagerNode::battery_callback, this, std::placeholders::_1));
  mavros_state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
    "mavros/state", 10, std::bind(&TelemetryManagerNode::mavros_state_callback, this, std::placeholders::_1));

  // setup publishers
  flight_status_pub_ = this->create_publisher<kestrel_msgs::msg::FlightStatus>("drone/flight_status", 10);
  system_health_pub_ = this->create_publisher<kestrel_msgs::msg::SystemHealth>("drone/system_health", 10);

  // set a timer to publish aggregated telemetry at 1hz
  timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&TelemetryManagerNode::publish_telemetry, this));

  RCLCPP_INFO(this->get_logger(), "TelemetryManagerNode initialized.");
}

void TelemetryManagerNode::battery_callback(const sensor_msgs::msg::BatteryState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_battery_state_ = *msg;
  has_battery_data_ = true;
}

void TelemetryManagerNode::mavros_state_callback(const mavros_msgs::msg::State::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_mavros_state_ = *msg;
  has_mavros_state_ = true;
}

float TelemetryManagerNode::get_cpu_temperature()
{
    // this function reads the cpu temperature from the jetson nano's virtual filesystem.
    std::ifstream temp_file("/sys/devices/virtual/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        float temp;
        temp_file >> temp;
        return temp / 1000.0f; // value is in millidegrees C
    }
    return -1.0f; // return an error value if the file can't be opened
}

float TelemetryManagerNode::get_cpu_usage()
{
    // this function reads and calculates cpu usage from /proc/stat
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) { return -1.0f; }

    std::string line;
    std::getline(stat_file, line);
    std::stringstream ss(line);
    std::string cpu_label;
    ss >> cpu_label;

    long long user, nice, system, idle, iowait, irq, softirq, steal;
    ss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    long long idle_time = idle + iowait;
    long long total_time = user + nice + system + idle + iowait + irq + softirq + steal;

    long long total_diff = total_time - prev_total_time_;
    long long idle_diff = idle_time - prev_idle_time_;

    prev_total_time_ = total_time;
    prev_idle_time_ = idle_time;

    if (total_diff == 0) { return 0.0f; }
    return 100.0f * (1.0f - static_cast<float>(idle_diff) / static_cast<float>(total_diff));
}

float TelemetryManagerNode::get_memory_usage()
{
    // this function reads and calculates memory usage from /proc/meminfo
    std::ifstream mem_file("/proc/meminfo");
    if (!mem_file.is_open()) { return -1.0f; }

    std::string line;
    long long mem_total = -1, mem_available = -1;
    while (std::getline(mem_file, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            std::stringstream ss(line);
            std::string label;
            ss >> label >> mem_total;
        }
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::stringstream ss(line);
            std::string label;
            ss >> label >> mem_available;
        }
    }

    if (mem_total > 0 && mem_available > 0) {
        return 100.0f * (1.0f - static_cast<float>(mem_available) / static_cast<float>(mem_total));
    }
    return -1.0f;
}

void TelemetryManagerNode::publish_telemetry()
{
  // create and populate the flight status message
  auto flight_status_msg = kestrel_msgs::msg::FlightStatus();
  flight_status_msg.header.stamp = this->get_clock()->now();

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (has_battery_data_) {
      flight_status_msg.battery_percent = latest_battery_state_.percentage * 100.0f;
    }
    if (has_mavros_state_) {
      flight_status_msg.armed = latest_mavros_state_.armed;
      // map the mavros string mode to our custom enum
      if (latest_mavros_state_.mode == "GUIDED") {
        flight_status_msg.flight_mode = kestrel_msgs::msg::FlightStatus::MODE_AUTO;
      } else if (latest_mavros_state_.mode == "STABILIZE" || latest_mavros_state_.mode == "ACRO") {
        flight_status_msg.flight_mode = kestrel_msgs::msg::FlightStatus::MODE_MANUAL;
      } else if (latest_mavros_state_.mode == "LAND") {
        flight_status_msg.flight_mode = kestrel_msgs::msg::FlightStatus::MODE_LANDING;
      } else {
        flight_status_msg.flight_mode = kestrel_msgs::msg::FlightStatus::MODE_UNKNOWN;
      }
    }
  }
  flight_status_pub_->publish(flight_status_msg);


  // create and populate the system health message
  auto system_health_msg = kestrel_msgs::msg::SystemHealth();
  system_health_msg.header.stamp = this->get_clock()->now();
  system_health_msg.status = kestrel_msgs::msg::SystemHealth::STATUS_OK; // default to ok

  // gather and add system metrics
  system_health_msg.temperature_celsius = get_cpu_temperature();
  system_health_msg.cpu_usage_percent = get_cpu_usage();
  system_health_msg.memory_usage_percent = get_memory_usage();
  
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (has_battery_data_) {
      diagnostic_msgs::msg::KeyValue battery_voltage;
      battery_voltage.key = "battery_voltage_v";
      battery_voltage.value = std::to_string(latest_battery_state_.voltage);
      system_health_msg.components.push_back(battery_voltage);
    }
  }
  system_health_pub_->publish(system_health_msg);
}

} // namespace kestrel_communication