#include "kestrel_control/frame_transformer.hpp"

namespace kestrel_control
{

FrameTransformer::FrameTransformer(const rclcpp::NodeOptions & options)
: Node("frame_transformer", options)
{
  initialize();
}

void FrameTransformer::initialize()
{
  // create publisher and subscriber
  local_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("odometry/local_pose", 10);
  gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
    "sensors/gps", 10, std::bind(&FrameTransformer::gps_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "FrameTransformer node initialized. Waiting for GPS fix to set origin.");
}

void FrameTransformer::gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  // check for a valid gps fix
  if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
    RCLCPP_WARN_ONCE(this->get_logger(), "waiting for valid gps fix...");
    return;
  }

  // use the first valid gps reading to set the origin of our local map
  if (!is_initialized_) {
    geo_converter_ = std::make_unique<GeographicLib::LocalCartesian>(msg->latitude, msg->longitude, msg->altitude);
    is_initialized_ = true;
    RCLCPP_INFO(this->get_logger(), "gps origin set to: lat=%.6f, lon=%.6f, alt=%.2f", msg->latitude, msg->longitude, msg->altitude);
    return;
  }

  // convert the current gps coordinates to local x, y, z
  double local_x, local_y, local_z;
  geo_converter_->Forward(msg->latitude, msg->longitude, msg->altitude, local_x, local_y, local_z);

  // create and publish the pose in the local frame
  auto pose_msg = geometry_msgs::msg::PoseStamped();
  pose_msg.header.stamp = msg->header.stamp;
  pose_msg.header.frame_id = "odom"; // or your odom frame name.. put whatever u want sabrina
  pose_msg.pose.position.x = local_x;
  pose_msg.pose.position.y = local_y;
  pose_msg.pose.position.z = local_z;
  pose_msg.pose.orientation.w = 1.0; // ASSUMING no orientation from gps

  local_pose_pub_->publish(pose_msg);
}

} // namespace kestrel_control

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kestrel_control::FrameTransformer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}