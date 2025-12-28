#include "waypoint_manager_node.hpp"

WayPointManagerNode::WayPointManagerNode() 
: rclcpp::Node("kestrel_waypoint"),
  current_waypoint_index_(0),
  waypoint_tolerance_(0.5),  // 0.5 meters tolerance
  has_current_pose_(false),
  has_goal_(false),
  has_path_(false)
{
    start_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "odometry/local_pose", 10, 
        std::bind(&WayPointManagerNode::currentPoseCallback, this, std::placeholders::_1));

    goal_sub_ = this->create_subscription<mavros_msgs::msg::PositionTarget>(
        "mavros/setpoint_raw/local", 10, 
        std::bind(&WayPointManagerNode::goalCallback, this, std::placeholders::_1));

    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "planning/path", 10, 
        std::bind(&WayPointManagerNode::pathCallback, this, std::placeholders::_1));
    
    position_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("planning/waypoint", 10);
    
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&WayPointManagerNode::updateWaypoint, this));
    
    RCLCPP_INFO(this->get_logger(), "Waypoint Manager Node initialized");
}

void WayPointManagerNode::currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    current_pose_ = *msg;
    has_current_pose_ = true;
}

void WayPointManagerNode::goalCallback(const mavros_msgs::msg::PositionTarget::SharedPtr msg)
{
    goal_position_ = *msg;
    has_goal_ = true;
}

void WayPointManagerNode::pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
    if (msg->poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty path");
        return;
    }
    
    current_path_ = *msg;
    has_path_ = true;
    current_waypoint_index_ = 0;  // Reset to start of new path
    
    RCLCPP_INFO(this->get_logger(), "Received new path with %zu waypoints", 
                current_path_.poses.size());
}

void WayPointManagerNode::updateWaypoint()
{
    if (!has_current_pose_ || !has_path_ || current_path_.poses.empty()) {
        return;
    }
    
    // Check if we've reached the current waypoint
    if (current_waypoint_index_ < current_path_.poses.size()) {
        const auto& current_waypoint = current_path_.poses[current_waypoint_index_];
        
        double distance = calculateDistance(current_pose_, current_waypoint);
        
        if (distance < waypoint_tolerance_) {
            // Move to next waypoint
            current_waypoint_index_++;
            
            if (current_waypoint_index_ >= current_path_.poses.size()) {
                RCLCPP_INFO(this->get_logger(), "Reached end of path");
                return;
            }
        }
        
        // Publish the current target waypoint
        publishCurrentWaypoint();
    }
}

void WayPointManagerNode::publishCurrentWaypoint()
{
    if (current_waypoint_index_ >= current_path_.poses.size()) {
        return;
    }
    
    const auto& waypoint_to_publish = current_path_.poses[current_waypoint_index_];
    
    geometry_msgs::msg::PoseStamped waypoint_msg = waypoint_to_publish;
    waypoint_msg.header.stamp = this->get_clock()->now();
    waypoint_msg.header.frame_id = current_path_.header.frame_id;
    
    position_pub_->publish(waypoint_msg);
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Published waypoint %zu: [%.2f, %.2f, %.2f]",
                 current_waypoint_index_,
                 waypoint_msg.pose.position.x,
                 waypoint_msg.pose.position.y,
                 waypoint_msg.pose.position.z);
}

double WayPointManagerNode::calculateDistance(
    const geometry_msgs::msg::PoseStamped& pose1,
    const geometry_msgs::msg::PoseStamped& pose2)
{
    double dx = pose1.pose.position.x - pose2.pose.position.x;
    double dy = pose1.pose.position.y - pose2.pose.position.y;
    double dz = pose1.pose.position.z - pose2.pose.position.z;
    
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Optional: Method to set waypoint tolerance dynamically
void WayPointManagerNode::setWaypointTolerance(double tolerance)
{
    waypoint_tolerance_ = tolerance;
    RCLCPP_INFO(this->get_logger(), "Waypoint tolerance set to %.2f meters", tolerance);
}

// Optional: Get current waypoint info
size_t WayPointManagerNode::getCurrentWaypointIndex() const
{
    return current_waypoint_index_;
}

bool WayPointManagerNode::hasReachedGoal() const
{
    return has_path_ && current_waypoint_index_ >= current_path_.poses.size();
}


#ifndef EXCLUDE_MAIN
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WayPointManagerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
#endif