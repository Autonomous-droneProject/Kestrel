#include "driver/generic_tof_driver_node.hpp"

namespace driver
{

GenericTofDriverNode::GenericTofDriverNode(const rclcpp::NodeOptions & options)
: Node("generic_tof_driver_node", options)
{
  initialize();
}

void GenericTofDriverNode::initialize()
{
  // declare parameters so they can be set from a launch file
  this->declare_parameter<std::string>("frame_id", "tof_link");
  this->declare_parameter<int>("num_rois", 16);

  // load the parameter values
  frame_id_ = this->get_parameter("frame_id").as_string();
  num_rois_ = this->get_parameter("num_rois").as_int();

  // create the publisher
  data_pub_ = this->create_publisher<kestrel_msgs::msg::Vl53l1xData>("tof_data", 10);

  // set a timer to call our read_data function at 20hz
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&GenericTofDriverNode::read_data, this));

  RCLCPP_INFO(this->get_logger(), "Generic ToF Driver initialized for frame '%s'", frame_id_.c_str());
}

void GenericTofDriverNode::read_data()
{
  // create the message we're going to publish
  auto msg = kestrel_msgs::msg::Vl53l1xData();
  msg.header.stamp = this->get_clock()->now();
  msg.header.frame_id = frame_id_;

    // ok this is where the actual hardware communication logic would go
    // for now we'll just simulate some data
  for (int i = 0; i < num_rois_; ++i)
  {
    // simulate a distance reading
    uint16_t simulated_distance = 1200 + (i * 10);
    msg.distance_mm.push_back(simulated_distance);

    // simulate a status for that reading
    msg.sensor_status.push_back(kestrel_msgs::msg::Vl53l1xData::SENSOR_STATUS_OK);
  }

  // publish the populated message
  data_pub_->publish(msg);
}

} // namespace driver

// this makes the node discoverable as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(driver::GenericTofDriverNode)