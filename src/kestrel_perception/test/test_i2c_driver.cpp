#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/sensor_reading.hpp"
#include "kestrel_perception/i2c_driver.hpp"
#include <memory>

class I2CDriverTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<kestrel_perception::I2CDriver>(rclcpp::NodeOptions());
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

protected:
  std::shared_ptr<kestrel_perception::I2CDriver> node_;
};

// test case to ensure the node can be created without crashing
TEST_F(I2CDriverTest, NodeSpins)
{
  // if the node was created successfully in setup, this test passes
  ASSERT_TRUE(node_ != nullptr);
}

// test case to check if the node publishes messages
TEST_F(I2CDriverTest, PublishesData)
{
  bool message_received = false;
  kestrel_msgs::msg::SensorReading received_msg;

  // create a temporary subscriber to listen for the message
  auto sub = node_->create_subscription<kestrel_msgs::msg::SensorReading>(
    "sensors/tof_front",
    10,
    [&](const kestrel_msgs::msg::SensorReading::SharedPtr msg) {
      message_received = true;
      received_msg = *msg;
    });

  // let the node spin for a short time to allow for publishing
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node_);

  // assert that a message was received
  EXPECT_TRUE(message_received);
  EXPECT_EQ(received_msg.sensor_type, "tof");
}