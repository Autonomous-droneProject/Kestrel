#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "kestrel_msgs/msg/CameraCommand.msg"
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
            sub_position = this->create_subscription<geometry_msgs::msg::PoseStamped>("odom", 5, std::bind(&DynamicCameraControlNode::update_position_callback, this, std::placeholders::_1));
            
            camera_cmd_pub_ = this->create_publisher<kestrel_msgs::msg::CameraCommand>("servo_camera_command", 5);

            /*
            drone_cmd_pub_ = this->create_publisher<std_msgs::msg::SomethingCommand>("drone_overshoot_command", 5);
            */

            //PID k value parameters for controllers
            this->declare_parameter<double>("camera_control_pid.pan.p", 1.2);
            this->declare_parameter<double>("camera_control_pid.pan.i", 0.1);
            this->declare_parameter<double>("camera_control_pid.pan.d", 0.05);

            this->declare_parameter<double>("camera_control_pid.tilt.p", 1.5);
            this->declare_parameter<double>("camera_control_pid.tilt.i", 0.1);
            this->declare_parameter<double>("camera_control_pid.tilt.d", 0.08);

            pan_p_ = this->get_parameter("camera_control_pid.pan.p").as_double();
            pan_i_ = this->get_parameter("camera_control_pid.pan.i").as_double();
            pan_d_ = this->get_parameter("camera_control_pid.pan.d").as_double();

            tilt_p_ = this->get_parameter("camera_control_pid.tilt.p").as_double();
            tilt_i_ = this->get_parameter("camera_control_pid.tilt.i").as_double();
            tilt_d_ = this->get_parameter("camera_control_pid.tilt.d").as_double();

            x_pid_controller.initPid(pan_p_, pan_i_, pan_d_); //Left right
            y_pid_controller.initPid(tilt_p_, tilt_i_, tilt_d_); //Up down

            prev_time = this->now(); //time when first called
        }
    
    private:
        void update_position_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
            const rclcpp::Duration max_dt = rclcpp::Duration::from_seconds(0.1);
            const rclcpp::Duration min_dt = rclcpp::Duration::from_seconds(0.001);

            rclcpp::Time curr_time = msg->header.stamp;
            rclcpp::Duration dt = curr_time-prev_time;
            prev_time = curr_time; //new previous

            //clamp time
            if(dt > max_dt)
            {
                dt = max_dt;
            } 
            else if(dt < min_dt)
            {
                dt = min_dt;
            }

            double dt_sec = dt.seconds();

            double x_err = 0.0 - msg->pose.position.x; //Center - object position x
            double y_err = 0.0 - msg->pose.position.y; //Center - object position y

            double x_angle = x_pid_controller.computeCommand(x_err, dt_sec);
            double y_angle = y_pid_controller.computeCommand(y_err, dt_sec);

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

            cam_msg_.pan = x_angle;
            cam_msg_.tilt = y_angle;

            /*
            p_msg_x_motor.data = static_cast<int16_t>(excess_x_angle);
            p_msg_y_motor.data = static_cast<int16_t>(excess_y_angle);
            */

            camera_cmd_pub_->publish(cam_msg_);

            /*
            pub_drone_x->publish(p_msg_x_motor);
            pub_drone_y->publish(p_msg_y_motor);
            */
        }
};

//Run node
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicCameraControlNode>());
    rclcpp::shutdown();
    return 0;
}
