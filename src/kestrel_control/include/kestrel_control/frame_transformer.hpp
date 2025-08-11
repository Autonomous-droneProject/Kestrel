#ifndef KESTREL_CONTROL__FRAME_TRANSFORMER_HPP_
#define KESTREL_CONTROL__FRAME_TRANSFORMER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "GeographicLib/LocalCartesian.hpp"
#include <memory>

namespace kestrel_control
{

class FrameTransformer : public rclcpp::Node
{
public:
  explicit FrameTransformer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

  // ros interfaces
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr local_pose_pub_;

  // geographiclib for gps to local frame conversion
  std::unique_ptr<GeographicLib::LocalCartesian> geo_converter_;
  bool is_initialized_ = false;
};

} // namespace kestrel_control
#endif // KESTREL_CONTROL__FRAME_TRANSFORMER_HPP_