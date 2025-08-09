#ifndef KESTREL_PERCEPTION__SENSOR_FUSION_NODE_HPP_
#define KESTREL_PERCEPTION__SENSOR_FUSION_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/sensor_array.hpp"
#include "kestrel_msgs/msg/obstacle_grid.hpp"
#include "kestrel_msgs/msg/proximity_alert.hpp"
#include "kestrel_perception/sensor_validator.hpp"
#include "kestrel_perception/spatial_grid.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <vector>
#include <mutex>

namespace kestrel_perception
{

class SensorFusionNode : public rclcpp::Node
{
public:
  explicit SensorFusionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void initialize();
  void sensor_array_callback(const kestrel_msgs::msg::SensorArray::SharedPtr msg);
  void update_loop();

  geometry_msgs::msg::Point convert_reading_to_point(const kestrel_msgs::msg::SensorReading & reading);
  // ^ converts sensor data into a 3d point, needs real logic for hardware

  // ros2 interfaces
  rclcpp::Subscription<kestrel_msgs::msg::SensorArray>::SharedPtr sensor_array_sub_;
  rclcpp::Publisher<kestrel_msgs::msg::ObstacleGrid>::SharedPtr obstacle_grid_pub_;
  rclcpp::Publisher<kestrel_msgs::msg::ProximityAlert>::SharedPtr proximity_alert_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // core stuff
  std::unique_ptr<SensorValidator> sensor_validator_;
  std::unique_ptr<SpatialGrid> spatial_grid_;

  // params
  double proximity_alert_distance_;
  std::string world_frame_id_;

  // latest data
  std::vector<kestrel_msgs::msg::SensorReading> latest_sensor_readings_;
  std::mutex data_mutex_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

} // namespace kestrel_perception

#endif // KESTREL_PERCEPTION__SENSOR_FUSION_NODE_HPP_
