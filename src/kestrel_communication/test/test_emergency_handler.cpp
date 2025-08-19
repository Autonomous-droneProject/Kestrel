#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/proximity_alert.hpp"
#include "kestrel_msgs/srv/trigger_emergency_stop.hpp"
#include "kestrel_communication/emergency_handler.hpp"
#include <memory>

class EmergencyHandlerTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<kestrel_communication::EmergencyHandlerNode>(rclcpp::NodeOptions());
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

protected:
  std::shared_ptr<kestrel_communication::EmergencyHandlerNode> node_;
};

// test case to check proximity alert trigger
TEST_F(EmergencyHandlerTest, TriggersOnProximityAlert)
{
  bool estop_called = false;
  auto pub = node_->create_publisher<kestrel_msgs::msg::ProximityAlert>("perception/proximity_alert", 10);

  auto service = node_->create_service<kestrel_msgs::srv::TriggerEmergencyStop>(
    "services/trigger_emergency_stop",
    [&](const std::shared_ptr<kestrel_msgs::srv::TriggerEmergencyStop::Request> request,
        std::shared_ptr<kestrel_msgs::srv::TriggerEmergencyStop::Response>) {
      estop_called = true;
      EXPECT_EQ(request->reason, "critical proximity alert");
    });

  auto alert_msg = std::make_unique<kestrel_msgs::msg::ProximityAlert>();
  alert_msg->distance = 0.1; // hard coding it waayyy below the default 0.5m critical distance

  pub->publish(std::move(alert_msg));
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node_);

  EXPECT_TRUE(estop_called);
}