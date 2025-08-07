#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/sensor_array.hpp"
#include "kestrel_msgs/msg/obstacle_grid.hpp"
#include "kestrel_perception/sensor_fusion_node.hpp"
#include <memory>

class SensorFusionTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<kestrel_perception::SensorFusionNode>(rclcpp::NodeOptions());
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

protected:
  std::shared_ptr<kestrel_perception::SensorFusionNode> node_;
};

// test case to ensure the node can be created without crashing
TEST_F(SensorFusionTest, NodeSpins)
{
  ASSERT_TRUE(node_ != nullptr);
}

// test case to check if sensor data is fused into the grid
TEST_F(SensorFusionTest, FusesSensorData)
{
  bool grid_received = false;
  kestrel_msgs::msg::ObstacleGrid received_grid;

  // create a publisher to send a fake sensor array message
  auto pub = node_->create_publisher<kestrel_msgs::msg::SensorArray>("sensors/fused_readings", 10);

  // create a subscriber to listen for the resulting obstacle grid
  auto sub = node_->create_subscription<kestrel_msgs::msg::ObstacleGrid>(
    "perception/obstacle_grid",
    10,
    [&](const kestrel_msgs::msg::ObstacleGrid::SharedPtr msg) {
      grid_received = true;
      received_grid = *msg;
    });

  // create a fake sensor reading
  auto sensor_msg = kestrel_msgs::msg::SensorReading();
  sensor_msg.header.stamp = node_->get_clock()->now();
  sensor_msg.header.frame_id = "base_link"; // use a frame that exists
  sensor_msg.sensor_type = "tof";
  sensor_msg.value = 1.0;

  auto array_msg = std::make_unique<kestrel_msgs::msg::SensorArray>();
  array_msg->readings.push_back(sensor_msg);

  // publish the fake data
  pub->publish(std::move(array_msg));

  // let the node spin to process the data
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  rclcpp::spin_some(node_);

  // assert that an obstacle grid was generated
  EXPECT_TRUE(grid_received);

  // check if any cell in the grid is marked as occupied
  bool obstacle_found = false;
  for (const auto& cell : received_grid.data) {
    if (cell == 100) { // 100 means occupied
      obstacle_found = true;
      break;
    }
  }
  EXPECT_TRUE(obstacle_found);
}