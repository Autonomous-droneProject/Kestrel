#ifndef WAYPOINT_MANAGER_NODE_HPP
#define WAYPOINT_MANAGER_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/position_target.hpp>
#include <nav_msgs/msg/path.hpp>
#include <cmath>
#include <vector>

class WayPointManagerNode : public rclcpp::Node
{
public:
    WayPointManagerNode();
    
    // Optional utility methods
    void setWaypointTolerance(double tolerance);
    size_t getCurrentWaypointIndex() const;
    bool hasReachedGoal() const;

private:
    // Callback methods
    void currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void goalCallback(const mavros_msgs::msg::PositionTarget::SharedPtr msg);
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg);
    
    // Core logic methods
    void updateWaypoint();
    void publishCurrentWaypoint();
    double calculateDistance(const geometry_msgs::msg::PoseStamped& pose1,
                           const geometry_msgs::msg::PoseStamped& pose2);
    
    // ROS2 subscribers and publishers
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr start_sub_;
    rclcpp::Subscription<mavros_msgs::msg::PositionTarget>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr position_pub_;
    
    // Timer for periodic waypoint updates
    rclcpp::TimerBase::SharedPtr timer_;
    
    // State variables
    geometry_msgs::msg::PoseStamped current_pose_;
    mavros_msgs::msg::PositionTarget goal_position_;
    nav_msgs::msg::Path current_path_;
    
    size_t current_waypoint_index_;
    double waypoint_tolerance_;
    
    // Status flags
    bool has_current_pose_;
    bool has_goal_;
    bool has_path_;
};

#endif // WAYPOINT_MANAGER_NODE_HPP