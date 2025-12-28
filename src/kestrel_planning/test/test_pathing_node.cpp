#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>

#define private public
#include "pathing_node.hpp"
#undef private

#include "kestrel_msgs/msg/obstacle_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mavros_msgs/msg/position_target.hpp"
#include "std_msgs/msg/empty.hpp"


#if defined(__linux__)
#include <time.h>
#endif

static inline double cpu_seconds_best_effort() {
#if defined(__linux__)
  struct timespec ts;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
  }
#endif
  return double(std::clock()) / double(CLOCKS_PER_SEC);
}

static inline double wall_seconds() {
  using clock = std::chrono::steady_clock;
  static const auto t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

static int getenv_int(const char* n, int d) {
  if (auto* s = std::getenv(n)) return std::atoi(s);
  return d;
}

template <class Fn>
static void bench(const char* label, int warmup, int iters, Fn&& fn) {
  for (int i = 0; i < warmup; ++i) fn();

  const double w0 = wall_seconds();
  const double c0 = cpu_seconds_best_effort();

  for (int i = 0; i < iters; ++i) fn();

  const double w1 = wall_seconds();
  const double c1 = cpu_seconds_best_effort();

  const double wall_total = (w1 - w0);
  const double cpu_total  = (c1 - c0);

  std::cout << "[PERF] " << label
            << " iters=" << iters
            << " wall_total_s=" << wall_total
            << " cpu_total_s=" << cpu_total
            << " wall_us/iter=" << (wall_total / double(iters)) * 1e6
            << " cpu_us/iter="  << (cpu_total  / double(iters)) * 1e6
            << std::endl;
}

class DStarPerfFixture : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    int argc = 0;
    char** argv = nullptr;
    rclcpp::init(argc, argv);

    // Silence logs to avoid dominating perf results.
    rcutils_logging_set_logger_level("kestrel_planning", RCUTILS_LOG_SEVERITY_FATAL);
    rcutils_logging_set_logger_level("rclcpp", RCUTILS_LOG_SEVERITY_FATAL);
  }

  static void TearDownTestSuite() {
    rclcpp::shutdown();
  }

    void spinFor(std::chrono::milliseconds duration)
    {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < duration) {
            exec_->spin_some();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

  void SetUp() override {
    node_ = std::make_shared<DStarNode>();

    exec_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    exec_->add_node(node_);
    pub_map_    = node_->create_publisher<kestrel_msgs::msg::ObstacleGrid>("perception/obstacle_grid", 10);
    pub_odom_   = node_->create_publisher<geometry_msgs::msg::PoseStamped>("odometry/local_pose", 10);
    pub_goal_   = node_->create_publisher<mavros_msgs::msg::PositionTarget>("mavros/setpoint_raw/local", 10);
    pub_replan_ = node_->create_publisher<std_msgs::msg::Empty>("planning/replan", 10);

    spinFor(std::chrono::milliseconds(300));
  }

  void TearDown() override {
    if (exec_) exec_->remove_node(node_);
    exec_.reset();
    node_.reset(); // triggers DStarNode dtor (should stop/join its planning thread)
  }

  // Helper: publish then drain callbacks for a short bounded time
  template <typename MsgT>
  void publish_and_drain(rclcpp::Publisher<MsgT>& pub, const MsgT& msg) {
    pub.publish(msg);

    spinFor(std::chrono::milliseconds(300));
  }

  std::shared_ptr<DStarNode> node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> exec_;

  rclcpp::Publisher<kestrel_msgs::msg::ObstacleGrid>::SharedPtr pub_map_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_odom_;
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr pub_goal_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub_replan_;
};

TEST_F(DStarPerfFixture, Perf_MapCallback_Throughput) {
  const int warmup = getenv_int("DSTAR_PERF_WARMUP", 200);
  const int iters  = getenv_int("DSTAR_PERF_ITERS", 2000);

  kestrel_msgs::msg::ObstacleGrid msg;
  // If your callback loops over grid data, set realistic sizes:
  // msg.dim_x = 200; msg.dim_y = 200; msg.dim_z = 10; msg.resolution = 0.05;
  // msg.data.resize(msg.dim_x * msg.dim_y * msg.dim_z);

  bench("mapCallback(pub+spin_some)", warmup, iters, [&] {
    publish_and_drain(*pub_map_, msg);
  });
}

TEST_F(DStarPerfFixture, Perf_OdomCallback_Throughput) {
  const int warmup = getenv_int("DSTAR_PERF_WARMUP", 200);
  const int iters  = getenv_int("DSTAR_PERF_ITERS", 5000);

  geometry_msgs::msg::PoseStamped msg;
  msg.header.frame_id = "map";
  msg.pose.position.x = 1.0;
  msg.pose.position.y = 2.0;
  msg.pose.orientation.w = 1.0;

  bench("odomCallback(pub+spin_some)", warmup, iters, [&] {
    publish_and_drain(*pub_odom_, msg);
  });
}

TEST_F(DStarPerfFixture, Perf_GoalCallback_Throughput) {
  const int warmup = getenv_int("DSTAR_PERF_WARMUP", 200);
  const int iters  = getenv_int("DSTAR_PERF_ITERS", 3000);

  mavros_msgs::msg::PositionTarget msg;
  msg.position.x = 5.0;
  msg.position.y = 6.0;
  msg.position.z = 0.0;

  bench("goalCallback(pub+spin_some)", warmup, iters, [&] {
    publish_and_drain(*pub_goal_, msg);
  });
}

TEST_F(DStarPerfFixture, Perf_ReplanCallback_Throughput) {
  const int warmup = getenv_int("DSTAR_PERF_WARMUP", 200);
  const int iters  = getenv_int("DSTAR_PERF_ITERS", 10000);

  std_msgs::msg::Empty msg;

  bench("replanCallback(pub+spin_some)", warmup, iters, [&] {
    publish_and_drain(*pub_replan_, msg);
  });
}