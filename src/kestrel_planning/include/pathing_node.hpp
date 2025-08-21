#ifndef PATHING_NODE_HPP
#define PATHING_NODE_HPP

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_msgs/msg/string.hpp"
#include <std_msgs/msg/empty.hpp>
#include "dstar_planner.hpp" 


enum class PlanningMode {
    NEW_PATH,
    REPLAN
};


class DStarNode : public rclcpp::Node {
public:
    DStarNode();
    ~DStarNode();

private:
    enum class PlannerState {
        IDLE,
        NO_MAP,
        AWAITING_GOAL,
        PLANNING,
        REPLANNING,
        SUCCESS,
        FAILURE
    };
    bool planning_request_ = false;
    std::atomic<PlannerState> current_state_;
    bool got_costmap_;
    bool got_start_;
    bool got_goal_;
    bool has_valid_path_;

    PlanningMode planning_mode_;

    //void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg);
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void odomCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void replanCallback(const std_msgs::msg::Empty::SharedPtr msg);

    //rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr start_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr replan_sub_;

    std::unique_ptr<DSTARLITE> planner_;
    std::mutex planner_mutex_;
    std::thread planning_thread_;
    std::condition_variable planning_cv_;

    void triggerPlanning();
    void triggerPlanningLocked_();
    void planningWorker();
    void updatePlannerState(PlannerState new_state);
    void publishPath(const std::vector<geometry_msgs::msg::PoseStamped> &waypoints);
    void publishStatus();

};

#endif // PATHING_NODE_HPP
