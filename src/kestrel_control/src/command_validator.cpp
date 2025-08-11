#include "kestrel_control/command_validator.hpp"
#include <cmath>

namespace kestrel_control
{

CommandValidator::CommandValidator(const rclcpp::NodeOptions & options)
: Node("command_validator", options)
{
  initialize();
}

void CommandValidator::initialize()
{
  // load speed safety parameters
  this->declare_parameter<double>("safety.max_linear_speed", 2.0); // m/s
  this->declare_parameter<double>("safety.max_angular_speed", 1.5); // rad/s
  max_linear_speed_ = this->get_parameter("safety.max_linear_speed").as_double();
  max_angular_speed_ = this->get_parameter("safety.max_angular_speed").as_double();

  // do note that this is a high level software geofence
  // a primary, more reliable hardware geofence should also be configured on the flight controller itself
  this->declare_parameter<double>("safety.geofence.min_x", -100.0);
  this->declare_parameter<double>("safety.geofence.max_x", 100.0);
  this->declare_parameter<double>("safety.geofence.min_y", -100.0);
  this->declare_parameter<double>("safety.geofence.max_y", 100.0);
  this->declare_parameter<double>("safety.geofence.min_z", 0.5);
  this->declare_parameter<double>("safety.geofence.max_z", 10.0);
  geofence_min_x_ = this->get_parameter("safety.geofence.min_x").as_double();
  geofence_max_x_ = this->get_parameter("safety.geofence.max_x").as_double();
  geofence_min_y_ = this->get_parameter("safety.geofence.min_y").as_double();
  geofence_max_y_ = this->get_parameter("safety.geofence.max_y").as_double();
  geofence_min_z_ = this->get_parameter("safety.geofence.min_z").as_double();
  geofence_max_z_ = this->get_parameter("safety.geofence.max_z").as_double();

  // setup publishers and subscribers
  safe_cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel/safe", 10);
  unsafe_cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel/unsafe", 10, std::bind(&CommandValidator::command_callback, this, std::placeholders::_1));
  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "odometry/local_pose", 10, std::bind(&CommandValidator::pose_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "CommandValidator node initialized.");
}

void CommandValidator::pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  current_pose_ = *msg;
  has_pose_ = true;
}

void CommandValidator::command_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  if (is_safe(*msg)) {
    // if the command is safe, pass it through to the validated topic
    safe_cmd_pub_->publish(*msg);
  } else {
    // if unsafe, log a warning and don't publish it
    RCLCPP_WARN(this->get_logger(), "unsafe command received and blocked.");
  }
}

bool CommandValidator::is_safe(const geometry_msgs::msg::Twist & command)
{
  // check linear speed
  double linear_speed = std::sqrt(
    std::pow(command.linear.x, 2) + std::pow(command.linear.y, 2) + std::pow(command.linear.z, 2)
  );
  if (linear_speed > max_linear_speed_) {
    RCLCPP_WARN(this->get_logger(), "command blocked: linear speed exceeded limit.");
    return false;
  }

  // check angular speed
  double angular_speed = std::sqrt(
    std::pow(command.angular.x, 2) + std::pow(command.angular.y, 2) + std::pow(command.angular.z, 2)
  );
  if (angular_speed > max_angular_speed_) {
    RCLCPP_WARN(this->get_logger(), "command blocked: angular speed exceeded limit.");
    return false;
  }

  // check geofence if we have a current position
  if (has_pose_) {
    const auto& pos = current_pose_.pose.position;
    if (pos.x < geofence_min_x_ || pos.x > geofence_max_x_ ||
        pos.y < geofence_min_y_ || pos.y > geofence_max_y_ ||
        pos.z < geofence_min_z_ || pos.z > geofence_max_z_)
    {
      RCLCPP_WARN(this->get_logger(), "command blocked: outside of geofence.");
      return false;
    }
  }

  return true;
}

} // namespace kestrel_control

// Run node
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<kestrel_control::CommandValidator>());
  rclcpp::shutdown();
  return 0;
}


