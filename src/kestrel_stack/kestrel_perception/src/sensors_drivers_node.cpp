#include "rclcpp/rclcpp.hpp"
#include "tca9548a/srv/select_channel.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "vl53l1x/vl53l1x.hpp" // From slaghuis/ROS2-VL53L1X package

#include <chrono>
#include <memory>
#include <vector>
#include <string>

using namespace std::chrono_literals;

class SensorsDriversNode : public rclcpp::Node
{
public:
  SensorsDriversNode() : Node("sensors_drivers_node"), current_sensor_index_(0)
  {
    // Declare parameters
    this->declare_parameter<int>("num_sensors", 8);
    this->declare_parameter<std::string>("i2c_bus", "/dev/i2c-1");
    this->declare_parameter<int>("vl53l1x_address", 0x29);
    this->declare_parameter<double>("publish_rate", 10.0);

    // Get parameters
    num_sensors_ = this->get_parameter("num_sensors").as_int();
    i2c_bus_ = this->get_parameter("i2c_bus").as_string();
    vl53l1x_address_ = this->get_parameter("vl53l1x_address").as_int();
    double publish_rate = this->get_parameter("publish_rate").as_double();

    if (num_sensors_ < 1 || num_sensors_ > 8)
    {
      RCLCPP_FATAL(this->get_logger(), "Number of sensors must be between 1 and 8.");
      rclcpp::shutdown();
      return;
    }

    // Create a client for the TCA9548A channel selection service
    tca_client_ = this->create_client<tca9548a::srv::SelectChannel>("/select_channel");
    while (!tca_client_->wait_for_service(1s))
    {
      if (!rclcpp::ok())
      {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
        rclcpp::shutdown();
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
    }

    // Create a publisher for each sensor
    for (int i = 0; i < num_sensors_; ++i)
    {
      std::string topic_name = "vl53l1x/range_" + std::to_string(i);
      publishers_.push_back(this->create_publisher<sensor_msgs::msg::Range>(topic_name, 10));
      RCLCPP_INFO(this->get_logger(), "Created publisher for sensor %d on topic %s", i, topic_name.c_str());
    }
    
    // Create the VL53L1X sensor object
    sensor_ = std::make_unique<Vl53l1x>();
    sensor_->i2c_bus_ = i2c_bus_; // Set the i2c bus from the parameter
    sensor_->i2c_address_ = vl53l1x_address_; // Set the sensor's I2C address

    // Create a timer to read and publish sensor data
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / (publish_rate * num_sensors_)),
        std::bind(&SensorsDriversNode::read_and_publish, this));
  }

private:
  void read_and_publish()
  {
    // Select the channel for the current sensor
    auto request = std::make_shared<tca9548a::srv::SelectChannel::Request>();
    request->channel = current_sensor_index_;

    auto future = tca_client_->async_send_request(request);
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to call service /select_channel for channel %d", current_sensor_index_);
      return;
    }

    auto result = future.get();
    if (!result->success)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to select channel %d: %s", current_sensor_index_, result->message.c_str());
      return;
    }

    // Initialize the sensor on the selected channel
    // Note: Re-initializing for every read might be slow.
    if (!sensor_->init())
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize VL53L1X on channel %d", current_sensor_index_);
        return;
    }
    sensor_->startRanging();

    // Read the distance
    int distance_mm = sensor_->getDistance();
    sensor_->stopRanging();

    if (distance_mm > 0)
    {
        // Publish the reading
        auto range_msg = std::make_unique<sensor_msgs::msg::Range>();
        range_msg->header.stamp = this->now();
        range_msg->header.frame_id = "vl53l1x_link_" + std::to_string(current_sensor_index_);
        range_msg->radiation_type = sensor_msgs::msg::Range::INFRARED;
        range_msg->field_of_view = 0.47; // Approximately 27 degrees
        range_msg->min_range = 0.04;     // 40mm
        range_msg->max_range = 4.0;      // 4m
        range_msg->range = static_cast<float>(distance_mm) / 1000.0f;
        publishers_[current_sensor_index_]->publish(std::move(range_msg));
    }


    // Move to the next sensor for the next callback
    current_sensor_index_ = (current_sensor_index_ + 1) % num_sensors_;
  }

  rclcpp::Client<tca9548a::srv::SelectChannel>::SharedPtr tca_client_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> publishers_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<Vl53l1x> sensor_;
  
  int num_sensors_;
  std::string i2c_bus_;
  int vl53l1x_address_;
  int current_sensor_index_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorsDriversNode>());
  rclcpp::shutdown();
  return 0;
}