/**
 * @file traj_controller.cpp
 * @author Rafeed Khan
 * @brief 3D Trajectory Controller implementation using Pure Pursuit
 *
 * No dynamic allocation, deterministic O(1) update time (amortized along a path)
 */

#include "kestrel_control/traj_controller.hpp"

namespace kestrel_control
{

void TrajectoryController::configure(const TrajConfig & cfg)
{
  cfg_ = cfg;

  // Sanitizing my configuration values
  cfg_.lookahead_dist = sanitizePositive(cfg_.lookahead_dist, 0.1f);
  cfg_.max_v = sanitizePositive(cfg_.max_v, 0.1f);
  cfg_.cruise_v = clamp(cfg_.cruise_v, 0.0f, cfg_.max_v);
  cfg_.slowdown_dist = sanitizePositive(cfg_.slowdown_dist, 0.0f);
  cfg_.pos_tol = sanitizePositive(cfg_.pos_tol, 0.01f);
  cfg_.yaw_tol = sanitizePositive(cfg_.yaw_tol, 0.01f);
  cfg_.advance_tol = sanitizePositive(cfg_.advance_tol, 0.01f);
  cfg_.min_v_near_goal = clamp(cfg_.min_v_near_goal, 0.0f, cfg_.max_v);

  reset();
}

void TrajectoryController::setTrajectory(const Waypoint3D * pts, size_t count)
{
  pts_ = pts;
  count_ = count;

  seg_i_ = 0;
  finished_ = !(pts_ && (count_ > 0));

  lookahead_x_ = 0.0f;
  lookahead_y_ = 0.0f;
  lookahead_z_ = 0.0f;
}

void TrajectoryController::clearTrajectory()
{
  pts_ = nullptr;
  count_ = 0;
  seg_i_ = 0;
  finished_ = true;
}

void TrajectoryController::reset()
{
  seg_i_ = 0;
  finished_ = !(pts_ && (count_ > 0));
  lookahead_x_ = 0.0f;
  lookahead_y_ = 0.0f;
  lookahead_z_ = 0.0f;
}

TrajectoryCommand TrajectoryController::update(const Pose3D & pose, float dt)
{
  TrajectoryCommand cmd{};

  // No trajectory loaded - nothing to do
  if (!hasTrajectory()) {
    cmd.finished = true;
    finished_ = true;
    return cmd;
  }

  // Rejecting the bad dt values (it prevents numerical issues)
  if (!(dt >= cfg_.min_dt) || !(dt <= cfg_.max_dt) || !isFinite(dt)) {
    // Then I return my last known state but DONT move
    cmd.x = pose.x;
    cmd.y = pose.y;
    cmd.z = pose.z;
    cmd.yaw = pose.yaw;
    cmd.lookahead_x = lookahead_x_;
    cmd.lookahead_y = lookahead_y_;
    cmd.lookahead_z = lookahead_z_;
    cmd.finished = finished_;
    return cmd;
  }

  // SINGLE WAYPOINT CASE
  // Degenerate case: just go to the single point
  if (count_ == 1) {
    const float dx = pts_[0].x - pose.x;
    const float dy = pts_[0].y - pose.y;
    const float dz = pts_[0].z - pose.z;
    const float dist = hypot3(dx, dy, dz);

    // Check if we've reached the goal
    if (dist <= cfg_.pos_tol) {
      cmd.x = pts_[0].x;
      cmd.y = pts_[0].y;
      cmd.z = pts_[0].z;
      cmd.vx = 0.0f;
      cmd.vy = 0.0f;
      cmd.vz = 0.0f;

      // Use waypoint yaw if specified, otherwise keep current
      if (pts_[0].has_yaw) {
        cmd.yaw = pts_[0].yaw;
      } else {
        cmd.yaw = pose.yaw;
      }

      cmd.lookahead_x = pts_[0].x;
      cmd.lookahead_y = pts_[0].y;
      cmd.lookahead_z = pts_[0].z;
      cmd.finished = true;
      finished_ = true;
      return cmd;
    }

    // Not at goal yet! So we compute velocity toward it
    lookahead_x_ = pts_[0].x;
    lookahead_y_ = pts_[0].y;
    lookahead_z_ = pts_[0].z;

    // Determine desired speed
    float v_des = cfg_.cruise_v;
    if (pts_[0].has_vel) {
      v_des = clamp(pts_[0].v, 0.0f, cfg_.max_v);
    }

    // Slowdown near goal
    if (cfg_.slowdown_dist > 0.0f) {
      const float scale = clamp(dist / cfg_.slowdown_dist, 0.0f, 1.0f);
      v_des = clamp(v_des * scale, cfg_.min_v_near_goal, cfg_.max_v);
    }

    // Velocity vector toward goal
    const float inv_dist = (dist > 1e-6f) ? (1.0f / dist) : 0.0f;
    cmd.vx = v_des * dx * inv_dist;
    cmd.vy = v_des * dy * inv_dist;
    cmd.vz = v_des * dz * inv_dist;

    // Target position is the lookahead (the goal itself ofc)
    cmd.x = pts_[0].x;
    cmd.y = pts_[0].y;
    cmd.z = pts_[0].z;

    // Yaw aka the face direction of travel (in XY plane)
    const float horiz_speed = std::sqrt(cmd.vx * cmd.vx + cmd.vy * cmd.vy);
    if (horiz_speed > 0.1f) {
      cmd.yaw = std::atan2(cmd.vy, cmd.vx);
    } else if (pts_[0].has_yaw) {
      cmd.yaw = pts_[0].yaw;
    } else {
      cmd.yaw = pose.yaw;  // maintain current yaw when nearly stationary
    }

    cmd.lookahead_x = lookahead_x_;
    cmd.lookahead_y = lookahead_y_;
    cmd.lookahead_z = lookahead_z_;
    cmd.finished = false;
    finished_ = false;
    return cmd;
  }

  // MULTI-WAYPOINT PATH
  // Advance segment index if we're close enough to the next waypoint
  advanceIfNeeded(pose);

  // Check if we've reached the final goal
  const Waypoint3D & last = pts_[count_ - 1];
  const float dx_goal = last.x - pose.x;
  const float dy_goal = last.y - pose.y;
  const float dz_goal = last.z - pose.z;
  const float dist_goal = hypot3(dx_goal, dy_goal, dz_goal);

  if (dist_goal <= cfg_.pos_tol && seg_i_ >= (count_ - 2)) {
    // We've reached the final waypoint
    cmd.x = last.x;
    cmd.y = last.y;
    cmd.z = last.z;
    cmd.vx = 0.0f;
    cmd.vy = 0.0f;
    cmd.vz = 0.0f;

    if (last.has_yaw) {
      cmd.yaw = last.yaw;
    } else {
      cmd.yaw = pose.yaw;
    }

    cmd.lookahead_x = last.x;
    cmd.lookahead_y = last.y;
    cmd.lookahead_z = last.z;
    cmd.finished = true;
    finished_ = true;
    return cmd;
  }

  // Compute lookahead point on the polyline
  float lx = 0.0f, ly = 0.0f, lz = 0.0f;
  computeLookaheadPoint3D(pose, &lx, &ly, &lz);
  lookahead_x_ = lx;
  lookahead_y_ = ly;
  lookahead_z_ = lz;

  // Direction and distance to lookahead
  const float dlx = lx - pose.x;
  const float dly = ly - pose.y;
  const float dlz = lz - pose.z;
  const float dist_lookahead = hypot3(dlx, dly, dlz);

  // Determine desired speed from current segment's end waypoint
  const Waypoint3D & seg_end = pts_[seg_i_ + 1];
  float v_des = cfg_.cruise_v;
  if (seg_end.has_vel) {
    v_des = seg_end.v;
  }
  v_des = clamp(v_des, 0.0f, cfg_.max_v);

  // Slowdown near the final goal
  if (cfg_.slowdown_dist > 0.0f) {
    const float scale = clamp(dist_goal / cfg_.slowdown_dist, 0.0f, 1.0f);
    v_des = clamp(v_des * scale, cfg_.min_v_near_goal, cfg_.max_v);
  }

  // Velocity vector toward lookahead point
  const float inv_dist = (dist_lookahead > 1e-6f) ? (1.0f / dist_lookahead) : 0.0f;
  cmd.vx = v_des * dlx * inv_dist;
  cmd.vy = v_des * dly * inv_dist;
  cmd.vz = v_des * dlz * inv_dist;

  // Target position is the lookahead point
  cmd.x = lx;
  cmd.y = ly;
  cmd.z = lz;

  // Yaw
  const float horiz_speed = std::sqrt(cmd.vx * cmd.vx + cmd.vy * cmd.vy);
  if (horiz_speed > 0.1f) {
    cmd.yaw = std::atan2(cmd.vy, cmd.vx);
  } else {
    cmd.yaw = pose.yaw;  // maintain current yaw when nearly stationary
  }

  cmd.lookahead_x = lx;
  cmd.lookahead_y = ly;
  cmd.lookahead_z = lz;
  cmd.finished = false;
  finished_ = false;
  return cmd;
}

void TrajectoryController::advanceIfNeeded(const Pose3D & pose)
{
  // If we're already at the last segment, don't advance!
  if (seg_i_ >= (count_ - 2)) {
    return;
  }

  // Check distance to the next waypoint
  const Waypoint3D & next = pts_[seg_i_ + 1];
  const float dx = next.x - pose.x;
  const float dy = next.y - pose.y;
  const float dz = next.z - pose.z;
  const float dist = hypot3(dx, dy, dz);

  // Advance if close enough
  if (dist <= cfg_.advance_tol) {
    seg_i_++;
    if (seg_i_ > (count_ - 2)) {
      seg_i_ = count_ - 2;
    }
  }
}

void TrajectoryController::computeLookaheadPoint3D(
  const Pose3D & pose, float * lx, float * ly, float * lz)
{
  // Default to last point if something goes wrong
  *lx = pts_[count_ - 1].x;
  *ly = pts_[count_ - 1].y;
  *lz = pts_[count_ - 1].z;

  // Starting segment
  size_t i = seg_i_;
  if (i >= (count_ - 1)) {
    i = count_ - 2;
  }

  // Lookahead distance to consume
  float remaining = cfg_.lookahead_dist;

  // Get current segment endpoints
  float ax = pts_[i].x;
  float ay = pts_[i].y;
  float az = pts_[i].z;
  float bx = pts_[i + 1].x;
  float by = pts_[i + 1].y;
  float bz = pts_[i + 1].z;

  // Segment direction vector
  const float sx = bx - ax;
  const float sy = by - ay;
  const float sz = bz - az;
  const float seg_len2 = sx * sx + sy * sy + sz * sz;

  // Project robot position onto current segment
  float px = ax;
  float py = ay;
  float pz = az;

  if (seg_len2 > 1e-12f) {
    // Compute projection parameter t in [0, 1]
    const float rx = pose.x - ax;
    const float ry = pose.y - ay;
    const float rz = pose.z - az;
    float t = (rx * sx + ry * sy + rz * sz) / seg_len2;
    t = clamp(t, 0.0f, 1.0f);

    // Projected point on segment
    px = ax + t * sx;
    py = ay + t * sy;
    pz = az + t * sz;

    // Distance from projection to segment end
    const float ex = bx - px;
    const float ey = by - py;
    const float ez = bz - pz;
    float seg_rem = hypot3(ex, ey, ez);

    // If lookahead fits within remainder of this segment then we interpolate here
    if (remaining <= seg_rem && seg_rem > 1e-9f) {
      const float u = remaining / seg_rem;
      *lx = px + u * ex;
      *ly = py + u * ey;
      *lz = pz + u * ez;
      return;
    }

    // Otherwise, consume this segment remainder and continue!
    remaining -= seg_rem;
  }

  // Walk forward along full segments until we consume remaining distance
  for (; i < (count_ - 2); i++) {
    ax = pts_[i + 1].x;
    ay = pts_[i + 1].y;
    az = pts_[i + 1].z;
    bx = pts_[i + 2].x;
    by = pts_[i + 2].y;
    bz = pts_[i + 2].z;

    const float dx = bx - ax;
    const float dy = by - ay;
    const float dz = bz - az;
    const float seg_len = hypot3(dx, dy, dz);

    if (seg_len < 1e-9f) {
      // Zero-length segment, skip
      continue;
    }

    if (remaining <= seg_len) {
      // Lookahead point is within this segment
      const float u = remaining / seg_len;
      *lx = ax + u * dx;
      *ly = ay + u * dy;
      *lz = az + u * dz;
      return;
    }

    remaining -= seg_len;
  }

  // If we run out of path, return last point (already set as default lol)
}

// UTILITY FUNCTIONS

float TrajectoryController::clamp(float x, float lo, float hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

bool TrajectoryController::isFinite(float x)
{
  // This filters out NaN and Inf
  return (x == x) && (x > -3.4e38f) && (x < 3.4e38f);
}

float TrajectoryController::wrapAngle(float a)
{
  // Wrap angle into [-pi, pi]
  constexpr float kPi = 3.14159265358979323846f;

  while (a > kPi) a -= 2.0f * kPi;
  while (a < -kPi) a += 2.0f * kPi;
  return a;
}

float TrajectoryController::hypot3(float x, float y, float z)
{
  return std::sqrt(x * x + y * y + z * z);
}

float TrajectoryController::sanitizePositive(float x, float fallback)
{
  if (!isFinite(x)) return fallback;
  if (x < 0.0f) return -x;
  return x;
}

}  // namespace kestrel_control
