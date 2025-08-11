#include "rclcpp.hpp"
#ifndef PID_HPP
#define PID_HPP

class PID
{
    
private:
    const rclcpp::Duration MIN_TIME = rclcpp::Duration::from_nanoseconds(1000000);
    const rclcpp::Duration MAX_TIME = rclcpp::Duration::from_nanoseconds(100000000);
    const float MIN_I = -1;
    const float MAX_I = 1;
    
    float error;
    float prevErr;
    rclcpp::Clock clock;
    rclcpp::Time prevTime;

public:
    float kp;
    float ki;
    float kd;
    float result;

    PID(float i_kp, float i_ki, float i_kd);
    float step(float offset);
    void resetErr();
};

#endif
