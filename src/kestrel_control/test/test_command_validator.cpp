#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "kestrel_control/command_validator.hpp"
#include <memory>

class CommandValidatorTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<kestrel_control::CommandValidator>(rclcpp::NodeOptions());
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

protected:
  std::shared_ptr<kestrel_control::CommandValidator> node_;
};

// test case to check that safe commands r allowed n stuff
TEST_F(CommandValidatorTest, AllowsSafeCommand)
{
  bool msg_received = false;
  auto pub = node_->create_publisher<geometry_msgs::msg::Twist>("cmd_vel/unsafe", 10);
  auto sub = node_->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel/safe", 10, [&](const geometry_msgs::msg::Twist::SharedPtr) { msg_received = true; });

  auto safe_cmd = std::make_unique<geometry_msgs::msg::Twist>();
  safe_cmd->linear.x = 1.0; // below the default limit of 2.0 m/s... this should make sense

  pub->publish(std::move(safe_cmd));
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node_);

  EXPECT_TRUE(msg_received);
}

// test case to check that unsafe commands are BBBBLOCKED!
TEST_F(CommandValidatorTest, BlocksUnsafeCommand)
{
  bool msg_received = false;
  auto pub = node_->create_publisher<geometry_msgs::msg::Twist>("cmd_vel/unsafe", 10);
  auto sub = node_->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel/safe", 10, [&](const geometry_msgs::msg::Twist::SharedPtr) { msg_received = true; });

  auto unsafe_cmd = std::make_unique<geometry_msgs::msg::Twist>();
  unsafe_cmd->linear.x = 5.0; // above the default limit of 2.0 m/s, also should make sense!

  pub->publish(std::move(unsafe_cmd));
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node_);

  EXPECT_FALSE(msg_received);
}