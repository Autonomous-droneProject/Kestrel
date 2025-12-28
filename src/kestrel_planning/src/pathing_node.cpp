#include "pathing_node.hpp"
#include "dstar_planner.hpp"

#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/empty.hpp>

#include "kestrel_msgs/msg/obstacle_grid.hpp"
#include "mavros_msgs/msg/position_target.hpp"


#include "kestrel_msgs/msg/obstacle_grid.hpp"
#include "mavros_msgs/msg/position_target.hpp"

#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <sstream>

using std::placeholders::_1;

DStarNode::DStarNode()
: rclcpp::Node("kestrel_planning"),
  planner_(nullptr),
  planning_request_(false),
  current_state_(PlannerState::IDLE),
  got_costmap_(false),
  got_start_(false),
  got_goal_(false),
  has_valid_path_(false),
  planning_mode_(PlanningMode::NEW_PATH)
{
    this->declare_parameter<int>("x_range", 100);
    this->declare_parameter<int>("y_range", 100);
    this->declare_parameter<int>("z_range", 100);

    int x = this->get_parameter("x_range").as_int();
    int y = this->get_parameter("y_range").as_int();
    int z = this->get_parameter("z_range").as_int();

    RCLCPP_INFO(this->get_logger(), "Creating DSTARLITE planner with ranges X:%d Y:%d Z:%d", x, y, z);

    planner_ = std::make_unique<DSTARLITE>(x, y, z);
    
    RCLCPP_INFO(this->get_logger(), "Successfully made node!");

    costmap_sub_ = this->create_subscription<kestrel_msgs::msg::ObstacleGrid>(
        "perception/obstacle_grid", 10, std::bind(&DStarNode::mapCallback, this, _1));

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();
    costmap_sub_ = this->create_subscription<kestrel_msgs::msg::ObstacleGrid>(
        "perception/obstacle_grid", 10, std::bind(&DStarNode::mapCallback, this, _1));

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();

    start_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "odometry/local_pose", qos, std::bind(&DStarNode::odomCallback, this, _1));
        "odometry/local_pose", qos, std::bind(&DStarNode::odomCallback, this, _1));

    goal_sub_ = this->create_subscription<mavros_msgs::msg::PositionTarget>(
        "mavros/setpoint_raw/local", 10, std::bind(&DStarNode::goalCallback, this, _1));
    goal_sub_ = this->create_subscription<mavros_msgs::msg::PositionTarget>(
        "mavros/setpoint_raw/local", 10, std::bind(&DStarNode::goalCallback, this, _1));

    replan_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "planning/replan", 10, std::bind(&DStarNode::replanCallback, this, _1));
        "planning/replan", 10, std::bind(&DStarNode::replanCallback, this, _1));

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planning/path", 10);
    status_pub_ = this->create_publisher<std_msgs::msg::String>("planning/planner_status", 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planning/path", 10);
    status_pub_ = this->create_publisher<std_msgs::msg::String>("planning/planner_status", 10);

    planning_thread_ = std::thread(&DStarNode::planningWorker, this);
}

DStarNode::~DStarNode()
{
    {
        std::lock_guard<std::mutex> lk(planner_mutex_);
        planning_request_ = false;
    }
    planning_cv_.notify_all();

    if (planning_thread_.joinable()) {
        planning_thread_.join();
    }
}

void DStarNode::mapCallback(const kestrel_msgs::msg::ObstacleGrid::SharedPtr msg)
void DStarNode::mapCallback(const kestrel_msgs::msg::ObstacleGrid::SharedPtr msg)
{
    std::lock_guard<std::mutex> lk(planner_mutex_);
    
    // Initialize costmap structure on first map callback
    if (!costmap_initialized_) {
        RCLCPP_INFO(this->get_logger(), "First map received - initializing costmap structure");
        planner_->initializeCostmap();
        costmap_initialized_ = true;
    }
    
    got_costmap_ = true;
    
    uint32_t width = msg->width;
    uint32_t height = msg->height;
    uint32_t depth = msg->depth;
    
    for (uint32_t z = 0; z < depth; ++z) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                size_t idx = z * (width * height) + y * width + x;
                
                if (idx < msg->data.size() && msg->data[idx] > 50) {
                    planner_->setOccupied(x, y, z);
                }
            }
        }
    }

    RCLCPP_INFO(this->get_logger(), "3D Map received: %u x %u x %u (res: %.3f)",
                width, height, depth, msg->resolution);

    if (has_valid_path_) {
        RCLCPP_INFO(this->get_logger(), "Map changed, triggering replan");
        has_valid_path_ = false;
    }
    //triggerPlanningLocked_();
}

void DStarNode::goalCallback(const mavros_msgs::msg::PositionTarget::SharedPtr msg)
void DStarNode::goalCallback(const mavros_msgs::msg::PositionTarget::SharedPtr msg)
{
    std::lock_guard<std::mutex> lk(planner_mutex_);
    
    planner_->setGoal(msg->position.x, msg->position.y, msg->position.z);
    
    planner_->setGoal(msg->position.x, msg->position.y, msg->position.z);
    got_goal_ = true;
    has_valid_path_ = false;
    
    
    RCLCPP_INFO(this->get_logger(), "Goal set: %.3f, %.3f, %.3f",
                msg->position.x, msg->position.y, msg->position.z);
    //triggerPlanningLocked_();
}

void DStarNode::odomCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lk(planner_mutex_);
    planner_->setStart(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    got_start_ = true;
    RCLCPP_INFO(this->get_logger(), "Start set: %.3f, %.3f, %.3f",
                msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    //triggerPlanningLocked_();
}

void DStarNode::replanCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg;
    std::lock_guard<std::mutex> lk(planner_mutex_);
    
    if (!costmap_initialized_) {
        RCLCPP_WARN(this->get_logger(), "Cannot plan - costmap not initialized yet");
        return;
    }
    
    if (has_valid_path_) {
        RCLCPP_INFO(this->get_logger(), "Replan requested - existing path will be updated");
        updatePlannerState(PlannerState::REPLANNING);
    } else {
        RCLCPP_INFO(this->get_logger(), "Replan requested - will compute new path");
    }
    
    if (got_costmap_ && got_start_ && got_goal_) {
        planner_->initialize();
        planning_request_ = true;
        planning_cv_.notify_one();
        RCLCPP_INFO(this->get_logger(), "Planning triggered successfully");
    } else {
        RCLCPP_WARN(this->get_logger(), "Cannot plan - missing required data (costmap=%d, start=%d, goal=%d)",
                   got_costmap_, got_start_, got_goal_);
        
        if (!got_costmap_) updatePlannerState(PlannerState::NO_MAP);
        else if (!got_goal_) updatePlannerState(PlannerState::AWAITING_GOAL);
        else if (!got_start_) updatePlannerState(PlannerState::IDLE);
    }
}

void DStarNode::triggerPlanning()
{
    std::lock_guard<std::mutex> lk(planner_mutex_);
    triggerPlanningLocked_();
}

void DStarNode::triggerPlanningLocked_()
{
    RCLCPP_INFO(this->get_logger(), "Planning check: costmap=%d start=%d goal=%d initialized=%d",
                got_costmap_, got_start_, got_goal_, costmap_initialized_);

    if (!costmap_initialized_) {
        updatePlannerState(PlannerState::NO_MAP);
        return;
    }

    if (got_costmap_ && got_start_ && got_goal_) {
        planner_->initialize();
        planning_request_ = true;
        RCLCPP_INFO(this->get_logger(), "Planning triggered");
        planning_cv_.notify_one();
    } else {
        if (!got_costmap_) updatePlannerState(PlannerState::NO_MAP);
        else if (!got_goal_) updatePlannerState(PlannerState::AWAITING_GOAL);
        else if (!got_start_) updatePlannerState(PlannerState::IDLE);
        else updatePlannerState(PlannerState::FAILURE);
    }
}

void DStarNode::planningWorker()
{
    while (rclcpp::ok()) {
        std::unique_lock<std::mutex> lk(planner_mutex_);
        planning_cv_.wait(lk, [this]{ return planning_request_ || !rclcpp::ok(); });

        if (!rclcpp::ok()) break;
        if (!planning_request_) continue;

        bool was_replanning = (planning_mode_ == PlanningMode::REPLAN);
        
        RCLCPP_INFO(this->get_logger(), "Planning worker running (%s)", 
                   was_replanning ? "REPLANNING" : "NEW_PATH");
        
        if (!was_replanning) {
            updatePlannerState(PlannerState::PLANNING);
        }

        try {
            int status = planner_->computeShortestPath();
            RCLCPP_INFO(this->get_logger(), "computeShortestPath returned: %d", status);

            if (status >= 0) {
                std::vector<geometry_msgs::msg::PoseStamped> waypoints;
                int extracted = planner_->extractPath(waypoints);
                RCLCPP_INFO(this->get_logger(), "extractPath returned: %d", extracted);

                if (extracted >= 0) {
                    nav_msgs::msg::Path path_msg;
                    path_msg.header.stamp = this->now();
                    path_msg.header.frame_id = "map";
                    path_msg.poses = std::move(waypoints);
                    path_pub_->publish(path_msg);
                    has_valid_path_ = true;
                    updatePlannerState(PlannerState::SUCCESS);
                    
                    if (was_replanning) {
                        RCLCPP_INFO(this->get_logger(), "Replan successful - path updated with %zu waypoints", 
                                   path_msg.poses.size());
                    } else {
                        RCLCPP_INFO(this->get_logger(), "New path computed with %zu waypoints", 
                                   path_msg.poses.size());
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "extractPath failed");
                    has_valid_path_ = false;
                    updatePlannerState(PlannerState::FAILURE);
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "computeShortestPath failed");
                has_valid_path_ = false;
                updatePlannerState(PlannerState::FAILURE);
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during planning: %s", e.what());
            has_valid_path_ = false;
            updatePlannerState(PlannerState::FAILURE);
        }

        planning_request_ = false;
        planning_mode_ = PlanningMode::NEW_PATH;
        lk.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void DStarNode::updatePlannerState(PlannerState new_state)
{
    current_state_.store(new_state);
    publishStatus();
}

void DStarNode::publishStatus()
{
    std_msgs::msg::String msg;
    switch (current_state_.load()) {
        case PlannerState::IDLE: msg.data = "IDLE"; break;
        case PlannerState::NO_MAP: msg.data = "NO_MAP"; break;
        case PlannerState::AWAITING_GOAL: msg.data = "AWAITING_GOAL"; break;
        case PlannerState::PLANNING: msg.data = "PLANNING"; break;
        case PlannerState::REPLANNING: msg.data = "REPLANNING"; break;
        case PlannerState::SUCCESS: msg.data = "SUCCESS"; break;
        case PlannerState::FAILURE: msg.data = "FAILURE"; break;
        default: msg.data = "UNKNOWN"; break;
    }
    status_pub_->publish(msg);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DStarNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
#endif