#ifndef KESTREL_PERCEPTION__SPATIAL_GRID_HPP_
#define KESTREL_PERCEPTION__SPATIAL_GRID_HPP_

#include "kestrel_msgs/msg/obstacle_grid.hpp"
#include "geometry_msgs/msg/point.hpp"
#include <vector>

namespace kestrel_perception
{
class SpatialGrid
{
public:
  SpatialGrid(int width, int height, int depth, double resolution);
  void add_obstacle(const geometry_msgs::msg::Point & point);
  void clear();
  kestrel_msgs::msg::ObstacleGrid to_ros_message();

private:
  int width_, height_, depth_;
  double resolution_;
  geometry_msgs::msg::Point origin_;
  std::vector<int8_t> data_;

  // constants for grid cell state
  static constexpr int8_t UNKNOWN = -1;
  static constexpr int8_t FREE = 0;
  static constexpr int8_t OCCUPIED = 100;
};
} // namespace kestrel_perception
#endif // KESTREL_PERCEPTION__SPATIAL_GRID_HPP_