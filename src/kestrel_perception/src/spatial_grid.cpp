#include "kestrel_perception/spatial_grid.hpp"

namespace kestrel_perception
{
SpatialGrid::SpatialGrid(int width, int height, int depth, double resolution)
: width_(width), height_(height), depth_(depth), resolution_(resolution)
{
  // the grids origin is at its center horizontally and at the bottom vertically lol
  origin_.x = - (width_ / 2.0) * resolution_;
  origin_.y = - (height_ / 2.0) * resolution_;
  origin_.z = 0.0;

  // resize the data vector and initialize all cells to unknown
  data_.resize(width_ * height_ * depth_);
  clear();
}

void SpatialGrid::add_obstacle(const geometry_msgs::msg::Point & point)
{
  // convert world coordinates to grid indices...
  int ix = static_cast<int>((point.x - origin_.x) / resolution_);
  int iy = static_cast<int>((point.y - origin_.y) / resolution_);
  int iz = static_cast<int>((point.z - origin_.z) / resolution_);

  // bounds check to make sure the point is inside the grid
  if (ix >= 0 && ix < width_ && iy >= 0 && iy < height_ && iz >= 0 && iz < depth_)
  {
    // convert 3d index to 1d index and mark as occupied
    int index = iz * (width_ * height_) + iy * width_ + ix;
    data_[index] = OCCUPIED;
  }
}

void SpatialGrid::clear()
{
  // HOPEFULLY this resets all cells to the unknown state
  std::fill(data_.begin(), data_.end(), UNKNOWN);
}

kestrel_msgs::msg::ObstacleGrid SpatialGrid::to_ros_message()
{
  kestrel_msgs::msg::ObstacleGrid msg;
  msg.resolution = resolution_;
  msg.width = width_;
  msg.height = height_;
  msg.depth = depth_;
  msg.origin.position = origin_;
  msg.origin.orientation.w = 1.0; // no rotation :D
  msg.data = data_;
  return msg;
}
} // namespace kestrel_perception