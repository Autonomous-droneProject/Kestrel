#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "control_toolbox/pid.hpp"
#include "dynamic_camera_control_node.hpp"

/*There was an attempt in here for me to use extra publishers to move the drone 
when the servos cannot move anymore, but that may need to be worked on.
All excess angle/motor control stuff is commented out so that it can be added in later when
we have to move the drone around
*/

class DynamicCameraControlNode : public rclcpp::Node
{
    public:
        DynamicCameraControlNode() : Node("camera_control")
        {
            sub_position = this->create_subscription<geometry_msgs::msg::Point>("position", 5, std::bind(&DynamicCameraControlNode::update_position_callback, this, std::placeholders::_1));
            
            pub_x_move = this->create_publisher<std_msgs::msg::UInt8>("servo_x_angle", 5);
            pub_y_move = this->create_publisher<std_msgs::msg::UInt8>("servo_y_angle", 5);

            /*
            pub_drone_x = this->create_publisher<std_msgs::msg::Int16>("drone_x_extra", 5);
            pub_drone_y = this->create_publisher<std_msgs::msg::Int16>("drone_y_extra", 5);
            */

            x_pid_controller.initPid(1.0, 0.0, 0.1); //Left right
            y_pid_controller.initPid(1.0, 0.0, 0.1); //Up down

            first_reading = true; //not read yet
            prev_time = this->now(); //time when first called
        }
    
    private:
        void update_position_callback(const geometry_msgs::msg::Point::SharedPtr msg)
        {
            rclcpp::Time curr_time = this->now();
            rclcpp::Duration dt = curr_time-prev_time;
            prev_time = curr_time; //new previous

            if(first_reading) //when you get the message you put in the coords
            {
                prev_x = msg->x;
                prev_y = msg->y;
                first_reading = false;
            }

            double x_err = msg->x - prev_x;
            double y_err = msg->y - prev_y;
            double x_angle = x_pid_controller.computeError(x_err, dt);
            double y_angle = y_pid_controller.computeError(y_err, dt);

            /*
            double excess_x_angle = 0.0;

            double excess_y_angle = 0.0;
            */

            if(x_angle > 180)
            {
                //excess_x_angle = x_angle - 180.0;
                x_angle = 180;
            } 
            else if(x_angle < 0)
            {
                //excess_x_angle = x_angle;
                x_angle = 0;
            }

            if(y_angle > 180)
            {
                //excess_y_angle = y_angle - 180.0;
                y_angle = 180;
            } 
            else if(y_angle < 0)
            {
                //excess_y_angle = y_angle;
                y_angle = 0;
            }

            p_msg_x.data = static_cast<uint8_t>(x_angle);
            p_msg_y.data = static_cast<uint8_t>(y_angle);

            /*
            p_msg_x_motor.data = static_cast<int16_t>(excess_x_angle);
            p_msg_y_motor.data = static_cast<int16_t>(excess_y_angle);
            */

            pub_x_move->publish(p_msg_x);
            pub_y_move->publish(p_msg_y);

            /*
            pub_drone_x->publish(p_msg_x_motor);
            pub_drone_y->publish(p_msg_y_motor);
            */
        }
};
