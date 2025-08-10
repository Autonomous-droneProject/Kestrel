#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "kestrel_msgs/msgs/camera_command.hpp"
#include "control_toolbox/pid.hpp"

#ifndef DYNAMIC_CAMERA_CONTROL_NODE_HPP_
#define DYNAMIC_CAMERA_CONTROL_NODE_HPP_

class DynamicCameraControlNode : public rclcpp::Node
{
    public:
        DynamicCameraControlNode();
    
    private:
        void update_position_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_position;

        double pan_p_, pan_i_, pan_d_;
        double tilt_p_, tilt_i_, tilt_d_;
        
        control_toolbox::Pid x_pid_controller;
        control_toolbox::Pid y_pid_controller;

        rclcpp::Time prev_time;

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

