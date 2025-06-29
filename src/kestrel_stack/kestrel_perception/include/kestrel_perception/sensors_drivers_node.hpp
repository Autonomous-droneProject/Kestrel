#ifndef SENSORS_DRIVERS_NODE_HPP_
#define SENSORS_DRIVERS_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tca9548a/srv/select_channel.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "vl53l1x/vl53l1x.hpp" // From slaghuis/ROS2-VL53L1X package

#include <memory>
#include <vector>
#include <string>

// Forward declaration for the Vl53l1x class
// This can sometimes help reduce compile times by not needing the full class definition
// in the header. However, since we use std::unique_ptr<Vl53l1x>, the full
// definition is required at the point of destruction, so we include vl53l1x.h above.
class Vl53l1x;

/**
 * @class SensorsDriversNode
 * @brief A ROS 2 node to manage and read from an array of VL53L1X time-of-flight
 * sensors connected via a TCA9548A I2C multiplexer.
 *
 * This node sequentially cycles through the I2C multiplexer's channels. For each channel,
 * it initializes a VL53L1X sensor, takes a distance measurement, and publishes it
 * on a unique topic corresponding to that sensor's channel.
 */
class SensorsDriversNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct a new Sensors Drivers Node object
   */
  SensorsDriversNode();

private:
  /**
   * @brief A timer-triggered callback function that performs one measurement cycle.
   *
   * This function selects the I2C channel for the current sensor, initializes the sensor,
   * reads the distance, publishes it, and then increments the sensor index for the next cycle.
   */
  void read_and_publish();

  // ROS 2 components
  rclcpp::Client<tca9548a::srv::SelectChannel>::SharedPtr tca_client_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> publishers_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Sensor driver object
  std::unique_ptr<Vl53l1x> sensor_;

  // Parameters and state variables
  int num_sensors_;
  std::string i2c_bus_;
  int vl53l1x_address_;
  int current_sensor_index_;
};

#endif // SENSORS_DRIVERS_NODE_HPP_