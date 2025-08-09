#ifndef DRIVER__GENERIC_TOF_DRIVER_NODE_HPP_
#define DRIVER__GENERIC_TOF_DRIVER_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/vl53l1x_data.hpp"
#include <vector>
#include <string>

namespace driver
{

class GenericTofDriverNode : public rclcpp::Node
{
public:
  explicit GenericTofDriverNode(const rclcpp::NodeOptions & options);

private:
  void initialize();
  void read_data();

  // publisher for the sensor data
  rclcpp::Publisher<kestrel_msgs::msg::Vl53l1xData>::SharedPtr data_pub_;

  // timer to trigger the read_data function periodically
  rclcpp::TimerBase::SharedPtr timer_;

  // parameters for the specific sensor instance
  std::string frame_id_;
  int num_rois_; // number of regions of interest for the sensor
};

} // namespace driver
#endif // DRIVER__GENERIC_TOF_DRIVER_NODE_HPP_