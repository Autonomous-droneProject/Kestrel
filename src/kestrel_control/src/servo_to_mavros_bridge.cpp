#include "rclcpp/rclcpp.hpp"
#include "kestrel_msgs/msg/camera_command.hpp"
#include "mavros_msgs/msg/mount_control.hpp"

namespace kestrel_control
{

class ServoToMavrosBridge : public rclcpp::Node
{
public:
    ServoToMavrosBridge() : Node("servo_to_mavros_bridge")
    {
        // subscribe to camera commands from dynamic_camera_control_node
        camera_sub_ = this->create_subscription<kestrel_msgs::msg::CameraCommand>(
            "servo_camera_command", 10,
            std::bind(&ServoToMavrosBridge::camera_callback, this, std::placeholders::_1));
        
        // publish dis to MAVROS mount control
        mount_pub_ = this->create_publisher<mavros_msgs::msg::MountControl>(
            "/mavros/mount_control/command", 10);
        
        RCLCPP_INFO(this->get_logger(), "Servo to MAVROS bridge initialized");
    }

private:
    void camera_callback(const kestrel_msgs::msg::CameraCommand::SharedPtr msg)
    {
        auto mount_msg = mavros_msgs::msg::MountControl();
        
        // Convert 0-180 degree range to -90 to +90 (typical gimbal range)
        mount_msg.pitch = msg->tilt - 90.0;  // Convert tilt to pitch
        mount_msg.yaw = msg->pan - 90.0;     // Convert pan to yaw
        mount_msg.roll = 0.0;                // Usually not used for gimbals
        
        // Mode 2 = MAVLINK_MSG_ID_MOUNT_CONTROL mode (angle control)
        mount_msg.mode = 2;
        
        mount_pub_->publish(mount_msg);
        
        RCLCPP_DEBUG(this->get_logger(), "Sent gimbal command: pitch=%.2f, yaw=%.2f", 
                     mount_msg.pitch, mount_msg.yaw);
    }
    
    rclcpp::Subscription<kestrel_msgs::msg::CameraCommand>::SharedPtr camera_sub_;
    rclcpp::Publisher<mavros_msgs::msg::MountControl>::SharedPtr mount_pub_;
};

} // namespace kestrel_control

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<kestrel_control::ServoToMavrosBridge>());
    rclcpp::shutdown();
    return 0;
}