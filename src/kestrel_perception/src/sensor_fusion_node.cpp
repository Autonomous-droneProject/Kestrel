#include "kestrel_perception/sensor_fusion_node.hpp"

namespace kestrel_perception
{

SensorFusionNode::SensorFusionNode(const rclcpp::NodeOptions & options)
: Node("sensor_fusion_node", options)
{
  initialize();
}

void SensorFusionNode::initialize()
{
  // set default params
  this->declare_parameter<double>("proximity_alert_distance", 1.0);
  this->declare_parameter<std::string>("world_frame_id", "odom");
  this->declare_parameter<double>("grid.resolution", 0.1);
  this->declare_parameter<int>("grid.width", 100);
  this->declare_parameter<int>("grid.height", 100);
  this->declare_parameter<int>("grid.depth", 20);

  // get param values
  proximity_alert_distance_ = this->get_parameter("proximity_alert_distance").as_double();
  world_frame_id_ = this->get_parameter("world_frame_id").as_string();

  // tf setup
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // init helpers
  sensor_validator_ = std::make_unique<SensorValidator>();
  spatial_grid_ = std::make_unique<SpatialGrid>(
    this->get_parameter("grid.width").as_int(),
    this->get_parameter("grid.height").as_int(),
    this->get_parameter("grid.depth").as_int(),
    this->get_parameter("grid.resolution").as_double()
  );

  // pub/sub/timer
  obstacle_grid_pub_ = this->create_publisher<kestrel_msgs::msg::ObstacleGrid>("perception/obstacle_grid", 10);
  proximity_alert_pub_ = this->create_publisher<kestrel_msgs::msg::ProximityAlert>("perception/proximity_alert", 10);

  sensor_array_sub_ = this->create_subscription<kestrel_msgs::msg::SensorArray>(
    "sensors/fused_readings",
    10,
    std::bind(&SensorFusionNode::sensor_array_callback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&SensorFusionNode::update_loop, this));

  RCLCPP_INFO(this->get_logger(), "SensorFusionNode has been initialized!");
}

void SensorFusionNode::sensor_array_callback(const kestrel_msgs::msg::SensorArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_sensor_readings_ = msg->readings;
}

void SensorFusionNode::update_loop()
{
  std::vector<kestrel_msgs::msg::SensorReading> readings_to_process;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (latest_sensor_readings_.empty())
    {
      return;
    }
    readings_to_process = latest_sensor_readings_;
    latest_sensor_readings_.clear();
  }

  spatial_grid_->clear();
  float min_obstacle_distance = std::numeric_limits<float>::max();

  for (const auto& reading : readings_to_process)
  {
    if (sensor_validator_->is_valid(reading))
    {
      geometry_msgs::msg::Point obstacle_point = convert_reading_to_point(reading);

      // skip invalid transform (0,0,0)
      if (obstacle_point.x != 0.0 || obstacle_point.y != 0.0 || obstacle_point.z != 0.0)
      {
        spatial_grid_->add_obstacle(obstacle_point);

        float distance = static_cast<float>(std::sqrt(
            std::pow(obstacle_point.x, 2) +
            std::pow(obstacle_point.y, 2) +
            std::pow(obstacle_point.z, 2)));

        if (distance < min_obstacle_distance)
        {
            min_obstacle_distance = distance;
        }
      }
    }
  }

  auto obstacle_grid_msg = spatial_grid_->to_ros_message();
  obstacle_grid_msg.header.stamp = this->get_clock()->now();
  obstacle_grid_msg.header.frame_id = world_frame_id_;
  obstacle_grid_pub_->publish(std::move(obstacle_grid_msg));

  if (min_obstacle_distance < proximity_alert_distance_)
  {
    auto alert_msg = kestrel_msgs::msg::ProximityAlert();
    alert_msg.header.stamp = this->get_clock()->now();
    alert_msg.header.frame_id = world_frame_id_;
    alert_msg.distance = min_obstacle_distance;
    alert_msg.severity = kestrel_msgs::msg::ProximityAlert::SEVERITY_HIGH;
    proximity_alert_pub_->publish(alert_msg);
  }
}

geometry_msgs::msg::Point SensorFusionNode::convert_reading_to_point(const kestrel_msgs::msg::SensorReading & reading)
{
  geometry_msgs::msg::PointStamped point_in_sensor_frame;
  point_in_sensor_frame.header = reading.header;
  // assume straight ahead on x-axis
  point_in_sensor_frame.point.x = reading.value;
  point_in_sensor_frame.point.y = 0.0;
  point_in_sensor_frame.point.z = 0.0;

  try
  {
    geometry_msgs::msg::Point point_in_world_frame = tf_buffer_->transform(point_in_sensor_frame, world_frame_id_).point;
    return point_in_world_frame;
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_WARN(
      this->get_logger(), "could not transform %s to %s: %s",
      reading.header.frame_id.c_str(), world_frame_id_.c_str(), ex.what());
  }

  return geometry_msgs::msg::Point(); // return zero if failed
}

} // namespace kestrel_perception

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<kestrel_perception::SensorFusionNode>());
  rclcpp::shutdown();
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kestrel_perception::SensorFusionNode)
