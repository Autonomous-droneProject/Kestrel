#include "kestrel_perception/sensor_validator.hpp"
#include <cmath>

namespace kestrel_perception
{
SensorValidator::SensorValidator()
{
  // setting some sensible default ranges
  min_range_ = 0.02; // 2 cm
  max_range_ = 4.0;  // 4 meters
}

bool SensorValidator::is_valid(const kestrel_msgs::msg::SensorReading & reading) const
{
  // check for non-numeric or infinite values
  if (std::isnan(reading.value) || std::isinf(reading.value))
  {
    return false;
  }

  // check if the reading is within the sensors operational range
  // I KNOW.. STUPID WAY TO DO THIS
  if (reading.value < min_range_ || reading.value > max_range_)
  {
    return false;
  }

  return true;
}
} // namespace kestrel_perception