/**
 * @file traj_controller_node.cpp
 * @author Rafeed Khan
 * @brief ROS2 Node wrapper for 3D Trajectory Controller.
 *
 * Subscribes to:
 *   - planning/path (nav_msgs/Path) - path from planner
 *   - odometry/local_pose (geometry_msgs/PoseStamped) 
 *   - current drone pose
 *
 * Publishes to:
 *   - trajectory/setpoint (mavros_msgs/PositionTarget) - position + velocity target
 */

#include <array>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "mavros_msgs/msg/position_target.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

#include "kestrel_control/traj_controller.hpp"

namespace kestrel_control
{

class TrajectoryControllerNode : public rclcpp::Node
{
public:
  explicit TrajectoryControllerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("traj_controller", options)
  {
    // Declarin my parameters with defaults
    this->declare_parameter("lookahead_dist", 1.0);
    this->declare_parameter("cruise_v", 1.5);
    this->declare_parameter("max_v", 2.0);
    this->declare_parameter("slowdown_dist", 2.0);
    this->declare_parameter("pos_tol", 0.3);
    this->declare_parameter("yaw_tol", 0.1);
    this->declare_parameter("advance_tol", 0.5);
    this->declare_parameter("min_v_near_goal", 0.1);
    this->declare_parameter("control_rate", 50.0);

    loadConfig();

    // Create subscribers
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "planning/path", 10,
      std::bind(&TrajectoryControllerNode::pathCallback, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "odometry/local_pose", 10,
      std::bind(&TrajectoryControllerNode::poseCallback, this, std::placeholders::_1));

    // Create publisher
    setpoint_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>(
      "trajectory/setpoint", 10);

    // Anddd my control loop timer
    double control_rate = this->get_parameter("control_rate").as_double();
    auto period = std::chrono::duration<double>(1.0 / control_rate);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&TrajectoryControllerNode::controlLoop, this));

    last_update_time_ = this->now();

    RCLCPP_INFO(this->get_logger(), "Trajectory Controller initialized");
    RCLCPP_INFO(this->get_logger(), "  lookahead_dist: %.2f m", cfg_.lookahead_dist);
    RCLCPP_INFO(this->get_logger(), "  cruise_v: %.2f m/s", cfg_.cruise_v);
    RCLCPP_INFO(this->get_logger(), "  max_v: %.2f m/s", cfg_.max_v);
    RCLCPP_INFO(this->get_logger(), "  control_rate: %.1f Hz", control_rate);
  }

private:
  void loadConfig()
  {
    cfg_.lookahead_dist = static_cast<float>(this->get_parameter("lookahead_dist").as_double());
    cfg_.cruise_v = static_cast<float>(this->get_parameter("cruise_v").as_double());
    cfg_.max_v = static_cast<float>(this->get_parameter("max_v").as_double());
    cfg_.slowdown_dist = static_cast<float>(this->get_parameter("slowdown_dist").as_double());
    cfg_.pos_tol = static_cast<float>(this->get_parameter("pos_tol").as_double());
    cfg_.yaw_tol = static_cast<float>(this->get_parameter("yaw_tol").as_double());
    cfg_.advance_tol = static_cast<float>(this->get_parameter("advance_tol").as_double());
    cfg_.min_v_near_goal = static_cast<float>(this->get_parameter("min_v_near_goal").as_double());

    controller_.configure(cfg_);
  }

  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
  {
    if (msg->poses.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty path");
      controller_.clearTrajectory();
      return;
    }

    // Convert nav_msgs::Path to internal waypoint format
    // Limit to MAX_WAYPOINTS to avoid overflow
    waypoint_count_ = std::min(msg->poses.size(), MAX_WAYPOINTS);

    for (size_t i = 0; i < waypoint_count_; i++) {
      const auto & pose = msg->poses[i].pose;
      waypoints_[i].x = static_cast<float>(pose.position.x);
      waypoints_[i].y = static_cast<float>(pose.position.y);
      waypoints_[i].z = static_cast<float>(pose.position.z);
      waypoints_[i].has_vel = 0;  // Path msg doesn't include velocity
      waypoints_[i].has_yaw = 0;  // Could extract from quaternion if needed though
    }

    // Set trajectory (borrows pointer, so no copy!!)
    controller_.setTrajectory(waypoints_.data(), waypoint_count_);

    RCLCPP_INFO(this->get_logger(), "Received path with %zu waypoints", waypoint_count_);
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    current_pose_.x = static_cast<float>(msg->pose.position.x);
    current_pose_.y = static_cast<float>(msg->pose.position.y);
    current_pose_.z = static_cast<float>(msg->pose.position.z);

    // Extract yaw from quaternion
    tf2::Quaternion q(
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    current_pose_.yaw = static_cast<float>(yaw);

    has_pose_ = true;
  }

  void controlLoop()
  {
    // Need pose before we can do anything
    if (!has_pose_) {
      return;
    }

    // Need a trajectory to follow
    if (!controller_.hasTrajectory()) {
      return;
    }

    // Already finished
    if (controller_.isFinished()) {
      return;
    }

    // Compute dt
    rclcpp::Time now = this->now();
    float dt = static_cast<float>((now - last_update_time_).seconds());
    last_update_time_ = now;

    // Update controller
    TrajectoryCommand cmd = controller_.update(current_pose_, dt);

    // Publish PositionTarget message
    mavros_msgs::msg::PositionTarget target;
    target.header.stamp = now;
    target.header.frame_id = "odom";
    target.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;

    // Type mask: use position + velocity + yaw
    // Ignore acceleration and yaw_rate
    target.type_mask =
      mavros_msgs::msg::PositionTarget::IGNORE_AFX |
      mavros_msgs::msg::PositionTarget::IGNORE_AFY |
      mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
      mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE;

    // Our lookahead point!
    target.position.x = cmd.x;
    target.position.y = cmd.y;
    target.position.z = cmd.z;

    // Velocity feedforward
    target.velocity.x = cmd.vx;
    target.velocity.y = cmd.vy;
    target.velocity.z = cmd.vz;

    // Yaw (face direction of travel)
    target.yaw = cmd.yaw;

    setpoint_pub_->publish(target);

    if (cmd.finished) {
      RCLCPP_INFO(this->get_logger(), "Trajectory following complete");
    }
  }

  // Fixed buffer so no dynamic allocation wow!
  static constexpr size_t MAX_WAYPOINTS = 1000;

  // ROS2 interfaces
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr setpoint_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Core controller
  TrajectoryController controller_;
  TrajConfig cfg_;

  // Pre allocated waypoint buffer!
  std::array<Waypoint3D, MAX_WAYPOINTS> waypoints_;
  size_t waypoint_count_ = 0;

  Pose3D current_pose_;
  bool has_pose_ = false;
  rclcpp::Time last_update_time_;
};

}  // namespace kestrel_control

// Main entry point
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<kestrel_control::TrajectoryControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
