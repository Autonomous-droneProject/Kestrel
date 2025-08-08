#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "kestrel_msgs/msg/CameraCommand.msg"
#include "control_toolbox/pid.hpp"

#ifndef DYNAMIC_CAMERA_CONTROL_NODE_HPP_
#define DYNAMIC_CAMERA_CONTROL_NODE_HPP_

class DynamicCameraControlNode : public rclcpp::Node
{
    public:
        DynamicCameraControlNode();
    
    private:
        void update_position_callback(const geometry_msgs::msg::Point::SharedPtr msg);

        rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr sub_position;
        
        control_toolbox::Pid x_pid_controller;
        control_toolbox::Pid y_pid_controller;

        rclcpp::Time prev_time;
        rclcpp::Duration dt;

        kestrel_msgs::msg::CameraCommand cam_msg_;

        /*
        std_msgs::msg::Int16 p_msg_x_motor;
        std_msgs::msg::Int16 p_msg_y_motor;
        */

        rclcpp::Publisher<kestrel_msgs::msg::CameraCommand>::SharedPtr camera_cmd_pub_;

        /*
        rclcpp::Publisher<std_msgs::msg::SomethingCommand>::SharedPtr drone_cmd_pub_;
        */
};

#endif //DYNAMIC_CAMERA_CONTROL_NODE_HPP_
