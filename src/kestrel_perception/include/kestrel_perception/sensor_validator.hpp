#ifndef KESTREL_PERCEPTION__SENSOR_VALIDATOR_HPP_
#define KESTREL_PERCEPTION__SENSOR_VALIDATOR_HPP_

#include "kestrel_msgs/msg/sensor_reading.hpp"
#include <limits>

namespace kestrel_perception
{
class SensorValidator
{
public:
  SensorValidator();
  bool is_valid(const kestrel_msgs::msg::SensorReading & reading) const;

private:
  // you can expand this to load sensor specific ranges from a config file if needed
  // for no now we dont really need that though...
  double min_range_;
  double max_range_;
};
} // namespace kestrel_perception
#endif // KESTREL_PERCEPTION__SENSOR_VALIDATOR_HPP_