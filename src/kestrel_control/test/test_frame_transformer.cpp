#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "kestrel_control/frame_transformer.hpp"
#include <memory>

class FrameTransformerTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<kestrel_control::FrameTransformer>(rclcpp::NodeOptions());
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

protected:
  std::shared_ptr<kestrel_control::FrameTransformer> node_;
};

// test case to check gps to local pose conversion
TEST_F(FrameTransformerTest, ConvertsGpsToLocal)
{
  bool pose_received = false;
  geometry_msgs::msg::PoseStamped received_pose;

  // create a publisher to send a fake gps message
  auto pub = node_->create_publisher<sensor_msgs::msg::NavSatFix>("sensors/gps", 10);

  // create a subscriber to listen for the resulting local pose
  auto sub = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
    "odometry/local_pose",
    10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      pose_received = true;
      received_pose = *msg;
    });

  // create a fake gps fix message
  auto gps_msg = std::make_unique<sensor_msgs::msg::NavSatFix>();
  gps_msg->header.stamp = node_->get_clock()->now();
  gps_msg->status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
  gps_msg->latitude = 47.397742;  // some sample coordinates
  gps_msg->longitude = 8.545594;
  gps_msg->altitude = 488.0;

  // publish the fake data twice, once to set the origin, once to get a pose
  pub->publish(std::move(gps_msg));
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  auto next_gps_msg = std::make_unique<sensor_msgs::msg::NavSatFix>(*gps_msg); // create a copy
  next_gps_msg->latitude += 0.0001; // move slightly
  pub->publish(std::move(next_gps_msg));

  // let the node spin to process the data
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node_);

  // assert that a pose was generated
  EXPECT_TRUE(pose_received);
  // check that the pose is not at the origin, WHICH MEANS A TRANSFORM HAPPENED!!
  EXPECT_NE(received_pose.pose.position.x, 0.0);
}