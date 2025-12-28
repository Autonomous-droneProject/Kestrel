#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <mavros_msgs/msg/position_target.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>

#include "waypoint_manager_node.hpp"

class WayPointManagerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rclcpp::init(0, nullptr);
        node_ = std::make_shared<WayPointManagerNode>();
        
        // Create test publisher nodes
        test_pub_node_ = std::make_shared<rclcpp::Node>("test_publishers");
        test_sub_node_ = std::make_shared<rclcpp::Node>("test_subscribers");
        
        // Create test publishers
        pose_pub_ = test_pub_node_->create_publisher<geometry_msgs::msg::PoseStamped>(
            "odometry/local_pose", 10);
        path_pub_ = test_pub_node_->create_publisher<nav_msgs::msg::Path>(
            "planning/path", 10);
        goal_pub_ = test_pub_node_->create_publisher<mavros_msgs::msg::PositionTarget>(
            "mavros/setpoint_raw/local", 10);
        
        // Create test subscriber
        waypoint_sub_ = test_sub_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "planning/waypoint", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                received_waypoints_.push_back(*msg);
            });
        
        // Allow time for publishers/subscribers to connect
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    void TearDown() override
    {
        node_.reset();
        test_pub_node_.reset();
        test_sub_node_.reset();
        pose_pub_.reset();
        path_pub_.reset();
        goal_pub_.reset();
        waypoint_sub_.reset();
        rclcpp::shutdown();
    }

    // Helper function to create a test path
    nav_msgs::msg::Path createTestPath(size_t num_waypoints, double spacing = 1.0)
    {
        nav_msgs::msg::Path path;
        path.header.frame_id = "map";
        path.header.stamp = rclcpp::Clock().now();
        
        for (size_t i = 0; i < num_waypoints; ++i) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position.x = i * spacing;
            pose.pose.position.y = i * spacing;
            pose.pose.position.z = 1.0;
            pose.pose.orientation.w = 1.0;
            path.poses.push_back(pose);
        }
        
        return path;
    }

    // Helper function to publish current pose
    void publishPose(double x, double y, double z)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = rclcpp::Clock().now();
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = z;
        pose.pose.orientation.w = 1.0;
        pose_pub_->publish(pose);
    }

    // Helper to spin node for a duration
    void spinFor(std::chrono::milliseconds duration)
    {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < duration) {
            rclcpp::spin_some(node_);
            rclcpp::spin_some(test_pub_node_);
            rclcpp::spin_some(test_sub_node_);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Get current memory usage in KB
    long getCurrentMemoryUsage()
    {
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line.substr(6));
                long memory_kb;
                iss >> memory_kb;
                return memory_kb;
            }
        }
        return -1;
    }

    // Get CPU usage
    double getCPUUsage()
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        double user_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
        double sys_time = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
        return user_time + sys_time;
    }

    std::shared_ptr<WayPointManagerNode> node_;
    std::shared_ptr<rclcpp::Node> test_pub_node_;
    std::shared_ptr<rclcpp::Node> test_sub_node_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr goal_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr waypoint_sub_;
    std::vector<geometry_msgs::msg::PoseStamped> received_waypoints_;
};

// ==================== PATH FUNCTIONALITY TESTS ====================

TEST_F(WayPointManagerTest, EmptyPathHandling)
{
    nav_msgs::msg::Path empty_path;
    empty_path.header.frame_id = "map";
    empty_path.header.stamp = rclcpp::Clock().now();
    
    path_pub_->publish(empty_path);
    spinFor(std::chrono::milliseconds(200));
    
    EXPECT_EQ(received_waypoints_.size(), 0);
}

TEST_F(WayPointManagerTest, MultipleWaypointPath)
{
    const size_t num_waypoints = 5;
    auto path = createTestPath(num_waypoints);
    path_pub_->publish(path);
    spinFor(std::chrono::milliseconds(100));
    
    publishPose(0.0, 0.0, 1.0);
    spinFor(std::chrono::milliseconds(300));
    
    EXPECT_GT(received_waypoints_.size(), 0);
    EXPECT_FALSE(node_->hasReachedGoal());
}

TEST_F(WayPointManagerTest, WaypointProgression)
{
    auto path = createTestPath(3, 1.0);
    path_pub_->publish(path);
    spinFor(std::chrono::milliseconds(100));
    
    // Start at first waypoint
    publishPose(0.0, 0.0, 1.0);
    spinFor(std::chrono::milliseconds(200));
    size_t initial_index = node_->getCurrentWaypointIndex();
    
    // Move to within tolerance of first waypoint
    publishPose(0.1, 0.1, 1.0);
    spinFor(std::chrono::milliseconds(200));
    
    // Should progress to next waypoint
    EXPECT_GE(node_->getCurrentWaypointIndex(), initial_index);
}

TEST_F(WayPointManagerTest, GoalReached)
{
    auto path = createTestPath(2, 0.5);
    path_pub_->publish(path);
    spinFor(std::chrono::milliseconds(100));
    
    // Move through all waypoints
    publishPose(0.0, 0.0, 1.0);
    spinFor(std::chrono::milliseconds(200));
    
    publishPose(0.5, 0.5, 1.0);
    spinFor(std::chrono::milliseconds(200));
    
    publishPose(0.5, 0.5, 1.0);
    spinFor(std::chrono::milliseconds(200));
    
    EXPECT_TRUE(node_->hasReachedGoal());
}

TEST_F(WayPointManagerTest, PathUpdate)
{
    auto path1 = createTestPath(3);
    path_pub_->publish(path1);
    spinFor(std::chrono::milliseconds(100));
    
    publishPose(0.5, 0.5, 1.0);
    spinFor(std::chrono::milliseconds(200));
    
    // Publish new path - should reset
    auto path2 = createTestPath(5);
    path_pub_->publish(path2);
    spinFor(std::chrono::milliseconds(100));
    
    EXPECT_EQ(node_->getCurrentWaypointIndex(), 0);
}

// ==================== RUNTIME PERFORMANCE TESTS ====================

TEST_F(WayPointManagerTest, SmallPathRuntime)
{
    const size_t num_waypoints = 10;
    auto path = createTestPath(num_waypoints);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    path_pub_->publish(path);
    publishPose(0.0, 0.0, 1.0);
    spinFor(std::chrono::milliseconds(500));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 1000) << "Processing 10 waypoints took too long";
    std::cout << "Small path (10 waypoints) runtime: " << duration.count() << " ms" << std::endl;
}

TEST_F(WayPointManagerTest, MediumPathRuntime)
{
    const size_t num_waypoints = 100;
    auto path = createTestPath(num_waypoints);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    path_pub_->publish(path);
    publishPose(0.0, 0.0, 1.0);
    spinFor(std::chrono::milliseconds(1000));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 2000) << "Processing 100 waypoints took too long";
    std::cout << "Medium path (100 waypoints) runtime: " << duration.count() << " ms" << std::endl;
}

TEST_F(WayPointManagerTest, LargePathRuntime)
{
    const size_t num_waypoints = 1000;
    auto path = createTestPath(num_waypoints);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    path_pub_->publish(path);
    publishPose(0.0, 0.0, 1.0);
    spinFor(std::chrono::milliseconds(2000));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 5000) << "Processing 1000 waypoints took too long";
    std::cout << "Large path (1000 waypoints) runtime: " << duration.count() << " ms" << std::endl;
}

// ==================== CPU UTILIZATION TESTS ====================

TEST_F(WayPointManagerTest, CPUUsageIdle)
{
    double cpu_before = getCPUUsage();
    
    // Let node idle
    spinFor(std::chrono::seconds(1));
    
    double cpu_after = getCPUUsage();
    double cpu_used = cpu_after - cpu_before;
    
    EXPECT_LT(cpu_used, 0.5) << "CPU usage too high during idle";
    std::cout << "CPU usage (idle): " << cpu_used << " seconds" << std::endl;
}

TEST_F(WayPointManagerTest, CPUUsageWithPath)
{
    auto path = createTestPath(100);
    
    double cpu_before = getCPUUsage();
    
    path_pub_->publish(path);
    publishPose(0.0, 0.0, 1.0);
    
    // Run for 1 second with active processing
    for (int i = 0; i < 100; ++i) {
        publishPose(i * 0.01, i * 0.01, 1.0);
        spinFor(std::chrono::milliseconds(10));
    }
    
    double cpu_after = getCPUUsage();
    double cpu_used = cpu_after - cpu_before;
    
    EXPECT_LT(cpu_used, 1.0) << "CPU usage too high with active path";
    std::cout << "CPU usage (with path): " << cpu_used << " seconds" << std::endl;
}

TEST_F(WayPointManagerTest, CPUUsageStressTest)
{
    auto path = createTestPath(1000);
    
    double cpu_before = getCPUUsage();
    
    path_pub_->publish(path);
    
    // Rapid pose updates
    for (int i = 0; i < 500; ++i) {
        publishPose(i * 0.002, i * 0.002, 1.0);
        rclcpp::spin_some(node_);
        rclcpp::spin_some(test_pub_node_);
        rclcpp::spin_some(test_sub_node_);
    }
    
    double cpu_after = getCPUUsage();
    double cpu_used = cpu_after - cpu_before;
    
    EXPECT_LT(cpu_used, 2.0) << "CPU usage too high under stress";
    std::cout << "CPU usage (stress test): " << cpu_used << " seconds" << std::endl;
}

// ==================== MEMORY UTILIZATION TESTS ====================

TEST_F(WayPointManagerTest, BaselineMemoryUsage)
{
    long memory = getCurrentMemoryUsage();
    
    EXPECT_GT(memory, 0) << "Failed to read memory usage";
    EXPECT_LT(memory, 100000) << "Baseline memory usage too high (>100MB)";
    std::cout << "Baseline memory usage: " << memory << " KB" << std::endl;
}

TEST_F(WayPointManagerTest, MemoryUsageWithSmallPath)
{
    long memory_before = getCurrentMemoryUsage();
    
    auto path = createTestPath(10);
    path_pub_->publish(path);
    spinFor(std::chrono::milliseconds(500));
    
    long memory_after = getCurrentMemoryUsage();
    long memory_increase = memory_after - memory_before;
    
    EXPECT_LT(memory_increase, 1000) << "Memory increase too large for small path";
    std::cout << "Memory increase (10 waypoints): " << memory_increase << " KB" << std::endl;
}

TEST_F(WayPointManagerTest, MemoryUsageWithLargePath)
{
    long memory_before = getCurrentMemoryUsage();
    
    auto path = createTestPath(1000);
    path_pub_->publish(path);
    spinFor(std::chrono::milliseconds(500));
    
    long memory_after = getCurrentMemoryUsage();
    long memory_increase = memory_after - memory_before;
    
    EXPECT_LT(memory_increase, 10000) << "Memory increase too large for large path (>10MB)";
    std::cout << "Memory increase (1000 waypoints): " << memory_increase << " KB" << std::endl;
}

TEST_F(WayPointManagerTest, MemoryLeakTest)
{
    long initial_memory = getCurrentMemoryUsage();
    
    // Create and process multiple paths
    for (int i = 0; i < 10; ++i) {
        auto path = createTestPath(100);
        path_pub_->publish(path);
        
        for (int j = 0; j < 10; ++j) {
            publishPose(j * 0.1, j * 0.1, 1.0);
            spinFor(std::chrono::milliseconds(10));
        }
    }
    
    // Force garbage collection
    spinFor(std::chrono::milliseconds(500));
    
    long final_memory = getCurrentMemoryUsage();
    long memory_diff = final_memory - initial_memory;
    
    EXPECT_LT(memory_diff, 5000) << "Potential memory leak detected (>5MB growth)";
    std::cout << "Memory difference after 10 iterations: " << memory_diff << " KB" << std::endl;
}

TEST_F(WayPointManagerTest, MemoryUsageUnderLoad)
{
    long memory_before = getCurrentMemoryUsage();
    
    // Process many paths in succession
    for (int i = 0; i < 100; ++i) {
        auto path = createTestPath(50);
        path_pub_->publish(path);
        publishPose(i * 0.05, i * 0.05, 1.0);
        rclcpp::spin_some(node_);
        rclcpp::spin_some(test_pub_node_);
        rclcpp::spin_some(test_sub_node_);
    }
    
    long memory_after = getCurrentMemoryUsage();
    long memory_increase = memory_after - memory_before;
    
    EXPECT_LT(memory_increase, 20000) << "Memory usage too high under load (>20MB)";
    std::cout << "Memory increase (100 path updates): " << memory_increase << " KB" << std::endl;
}

// ==================== COMBINED PERFORMANCE TEST ====================

TEST_F(WayPointManagerTest, OverallPerformanceProfile)
{
    std::cout << "\n=== Overall Performance Profile ===" << std::endl;
    
    long memory_start = getCurrentMemoryUsage();
    double cpu_start = getCPUUsage();
    auto time_start = std::chrono::high_resolution_clock::now();
    
    // Simulate realistic usage
    auto path = createTestPath(200);
    path_pub_->publish(path);
    
    for (int i = 0; i < 200; ++i) {
        publishPose(i * 0.01, i * 0.01, 1.0);
        spinFor(std::chrono::milliseconds(50));
    }
    
    auto time_end = std::chrono::high_resolution_clock::now();
    double cpu_end = getCPUUsage();
    long memory_end = getCurrentMemoryUsage();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    double cpu_used = cpu_end - cpu_start;
    long memory_used = memory_end - memory_start;
    
    std::cout << "Total runtime: " << duration.count() << " ms" << std::endl;
    std::cout << "CPU time used: " << cpu_used << " seconds" << std::endl;
    std::cout << "Memory used: " << memory_used << " KB" << std::endl;
    std::cout << "Waypoints processed: " << node_->getCurrentWaypointIndex() + 1 << std::endl;
    
    EXPECT_LT(duration.count(), 15000) << "Overall runtime too high";
    EXPECT_LT(cpu_used, 3.0) << "Overall CPU usage too high";
    EXPECT_LT(memory_used, 15000) << "Overall memory usage too high";
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}