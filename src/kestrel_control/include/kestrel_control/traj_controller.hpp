/**
 * @file traj_controller.hpp
 * @brief 3D Trajectory Controller for drone path following using Pure Pursuit.
 *
 * Lightweight trajectory follower adapted from 2D tank-drive controller.
 * No dynamic allocation, deterministic O(1) update time (amortized along a path).
 *
 * This is a Pure-Pursuit-style "carrot" tracker for 3D:
 *  - You give it an array of waypoints (x, y, z) with optional velocity/yaw.
 *  - Each update() returns a target position + velocity vector toward a lookahead point.
 *
 * Intended use:
 *  - The ROS2 node wrapper converts output to mavros_msgs::PositionTarget.
 *  - ArduPilot/PX4 flight controller handles low-level position/velocity control.
 */

#ifndef KESTREL_CONTROL__TRAJ_CONTROLLER_HPP_
#define KESTREL_CONTROL__TRAJ_CONTROLLER_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kestrel_control
{

/**
 * @brief 3D pose with position and yaw orientation.
 */
struct Pose3D
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float yaw = 0.0f;  // radians
};

/**
 * @brief 3D waypoint with optional velocity and yaw.
 */
struct Waypoint3D
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  // Optional per-waypoint desired linear velocity (m/s)
  // If has_vel == 0, we use cfg.cruise_v instead
  float v = 0.0f;
  uint8_t has_vel = 0;

  // Optional target yaw at this waypoint
  float yaw = 0.0f;  // radians
  uint8_t has_yaw = 0;
};

/**
 * @brief Output command from trajectory controller.
 */
struct TrajectoryCommand
{
  // Target position (lookahead point or interpolated position)
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  // Velocity hints toward lookahead (for feedforward)
  float vx = 0.0f;
  float vy = 0.0f;
  float vz = 0.0f;

  // Target yaw (face direction of travel by default)
  float yaw = 0.0f;

  // Debug: actual lookahead point in world frame
  float lookahead_x = 0.0f;
  float lookahead_y = 0.0f;
  float lookahead_z = 0.0f;

  // True when trajectory is complete
  bool finished = false;
};

/**
 * @brief Configuration parameters for trajectory controller.
 */
struct TrajConfig
{
  // Pure pursuit lookahead distance (meters)
  // Larger = smoother but cuts corners
  float lookahead_dist = 1.0f;

  // Cruise speed if waypoints don't specify velocity
  float cruise_v = 1.5f;

  // Output velocity clamp
  float max_v = 2.0f;

  // Slow down as we approach the final point
  float slowdown_dist = 2.0f;  // meters, 0 disables slowdown
  float min_v_near_goal = 0.1f;  // minimum creep speed near goal

  // Goal tolerances
  float pos_tol = 0.3f;  // meters - position tolerance for "reached"
  float yaw_tol = 0.1f;  // radians - yaw tolerance for final heading

  // "Progress" logic: how close before we advance along the path
  float advance_tol = 0.5f;  // meters

  // dt guards (reject bad timing values)
  float min_dt = 1e-6f;
  float max_dt = 0.25f;
};

/**
 * @brief 3D Pure Pursuit Trajectory Controller.
 *
 * O(1) amortized update time - maintains segment index to avoid searching
 * entire path each update. No dynamic allocation during update().
 */
class TrajectoryController
{
public:
  TrajectoryController() = default;
  explicit TrajectoryController(const TrajConfig & cfg) { configure(cfg); }

  /**
   * @brief Configure controller parameters.
   * @param cfg Configuration struct (values are sanitized internally).
   */
  void configure(const TrajConfig & cfg);

  /**
   * @brief Get current configuration.
   */
  const TrajConfig & config() const { return cfg_; }

  /**
   * @brief Set trajectory to follow.
   *
   * Does NOT copy - borrows pointer. Caller must keep waypoints alive.
   * This allows zero-allocation operation with pre-allocated buffers.
   *
   * @param pts Pointer to waypoint array.
   * @param count Number of waypoints.
   */
  void setTrajectory(const Waypoint3D * pts, size_t count);

  /**
   * @brief Clear current trajectory.
   */
  void clearTrajectory();

  /**
   * @brief Check if a trajectory is loaded.
   */
  bool hasTrajectory() const { return (pts_ != nullptr) && (count_ > 0); }

  /**
   * @brief Reset progress along current trajectory (start from beginning).
   */
  void reset();

  /**
   * @brief Main update: compute command for this timestep.
   *
   * @param pose Current drone pose.
   * @param dt Time since last update (seconds).
   * @return TrajectoryCommand with target position, velocity, and yaw.
   */
  TrajectoryCommand update(const Pose3D & pose, float dt);

  /**
   * @brief Check if trajectory following is complete.
   */
  bool isFinished() const { return finished_; }

  /**
   * @brief Get current segment index (for debugging).
   */
  size_t getCurrentSegment() const { return seg_i_; }

private:
  /**
   * @brief Compute lookahead point along the polyline path.
   *
   * Walks forward from current segment by lookahead_dist.
   * O(1) amortized since we start from current segment, not beginning.
   */
  void computeLookaheadPoint3D(const Pose3D & pose, float * lx, float * ly, float * lz);

  /**
   * @brief Advance segment index when close enough to next waypoint.
   */
  void advanceIfNeeded(const Pose3D & pose);

  // Utility functions
  static float clamp(float x, float lo, float hi);
  static bool isFinite(float x);
  static float wrapAngle(float a);
  static float hypot3(float x, float y, float z);
  static float sanitizePositive(float x, float fallback);

  // Configuration
  TrajConfig cfg_{};

  // Trajectory (borrowed, not owned)
  const Waypoint3D * pts_ = nullptr;
  size_t count_ = 0;

  // Path progress tracking (enables O(1) amortized updates)
  size_t seg_i_ = 0;  // current segment start index (segment is pts[i] -> pts[i+1])
  bool finished_ = true;

  // Cached lookahead point (for debugging/visualization)
  float lookahead_x_ = 0.0f;
  float lookahead_y_ = 0.0f;
  float lookahead_z_ = 0.0f;
};

}  // namespace kestrel_control

#endif  // KESTREL_CONTROL__TRAJ_CONTROLLER_HPP_
