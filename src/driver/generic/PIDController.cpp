#include "rclcpp/rclcpp.hpp"
#include "PIDController.hpp"

PID::PID(float i_kp, float i_ki, float i_kd)
{
    kp = i_kp;
    ki = i_ki;
    kd = i_kd;
    error = 0.0f;
    prevErr = 0.0f;
    clock(RCL_ROS_TIME);
    prevTime = clock.now();
    result = 0.0f;
}

float PID::step(float offset)
{
    float P = 0.0f;
    float I = 0.0f;
    float D = 0.0f;

    rclcpp::Time currTime = clock.now();
    rclcpp::Duration dt = currTime - prevTime;

    if(dt < MIN_TIME)
    {
        dt = MIN_TIME;
    } 
    else if(dt > MAX_TIME)
    {
        dt = MAX_TIME;
    }

    error = offset;
    P = error*kp;

    I += ki*error*dt;

    if(I < MIN_I)
    {
        I = MIN_I;
    }
    else if(I > MAX_I)
    {
        I = MAX_I;
    }

    D = (error-prevErr)/dt;

    result = P + I + D;
    prevErr = error;
    prevTime = currTime;
    return result;
}

void PID::resetErr()
{
    prevErr = 0.0f;
}