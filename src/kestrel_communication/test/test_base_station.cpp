#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "kestrel_msgs/srv/set_flight_mode.hpp"
#include "kestrel_communication/base_station.hpp"
#include <memory>

class BaseStationTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<kestrel_communication::BaseStationNode>(rclcpp::NodeOptions());
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

protected:
  std::shared_ptr<kestrel_communication::BaseStationNode> node_;
};

// test case to check command processing
TEST_F(BaseStationTest, ProcessesSetModeCommand)
{
  bool service_called = false;
  auto pub = node_->create_publisher<std_msgs::msg::String>("gcs/command", 10);

  // ok so im basically creating a dummy service server to check if the client call was made
  // its a little jank but it works
  auto service = node_->create_service<kestrel_msgs::srv::SetFlightMode>(
    "services/set_flight_mode",
    [&](const std::shared_ptr<kestrel_msgs::srv::SetFlightMode::Request> request,
        std::shared_ptr<kestrel_msgs::srv::SetFlightMode::Response>) {
      service_called = true;
      EXPECT_EQ(request->flight_mode, kestrel_msgs::msg::FlightStatus::MODE_AUTO);
    });

  auto command_msg = std::make_unique<std_msgs::msg::String>();
  command_msg->data = "SET_MODE:AUTO";

  pub->publish(std::move(command_msg));
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node_);

  EXPECT_TRUE(service_called);
}