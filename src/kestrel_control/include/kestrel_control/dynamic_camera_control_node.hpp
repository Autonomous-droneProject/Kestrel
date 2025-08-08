#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "control_toolbox/pid.hpp"

class DynamicCameraControlNode : public rclcpp::Node
{
    public:
        DynamicCameraControlNode();
    
    private:
        void update_position_callback(const geometry_msgs::msg::Point::SharedPtr msg);

        rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr sub_position;
        
        control_toolbox::Pid x_pid_controller;
        control_toolbox::Pid y_pid_controller;

        rclcpp::Duration prev_time;

        double prev_x;
        double prev_y;
        bool first_reading;
        rclcpp::Duration dt;

        std_msgs::msg::UInt8 p_msg_x;
        std_msgs::msg::UInt8 p_msg_y;

        /*
        std_msgs::msg::Int16 p_msg_x_motor;
        std_msgs::msg::Int16 p_msg_y_motor;
        */

        rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_x_move;
        rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_y_move;

        /*
        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr pub_drone_x;
        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr pub_drone_y;
        */
};
