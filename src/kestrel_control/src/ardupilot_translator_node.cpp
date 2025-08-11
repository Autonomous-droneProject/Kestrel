#include "kestrel_control/ardupilot_translator_node.hpp"

namespace kestrel_control
{

    ArduPilotTranslatorNode::ArduPilotTranslatorNode(const rclcpp::NodeOptions &options)
        : Node("ardupilot_translator_node", options)
    {
        initialize();
    }

    void ArduPilotTranslatorNode::initialize()
    {
        // setup publisher and subscriber
        target_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>("mavros/setpoint_raw/local", 10);
        waypoint_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "planning/waypoint", 10, std::bind(&ArduPilotTranslatorNode::waypoint_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "ArduPilotTranslatorNode initialized.");
    }

    void ArduPilotTranslatorNode::waypoint_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // create the mavros message
        auto target_msg = mavros_msgs::msg::PositionTarget();
        target_msg.header.stamp = this->get_clock()->now();
        target_msg.header.frame_id = "odom"; // assumes waypoints are in the odom frame

        // this tells the flight controller we are sending position commands
        target_msg.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;

        // ok so this bitmask tells the flight controller to ignore velocity, acceleration, and yaw rate
        // and ONLY pay attention to the position data we are sending
        target_msg.type_mask = mavros_msgs::msg::PositionTarget::IGNORE_VX |
                               mavros_msgs::msg::PositionTarget::IGNORE_VY |
                               mavros_msgs::msg::PositionTarget::IGNORE_VZ |
                               mavros_msgs::msg::PositionTarget::IGNORE_AFX |
                               mavros_msgs::msg::PositionTarget::IGNORE_AFY |
                               mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
                               mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE;

        // copy the position from the high-level waypoint to the low-level command
        target_msg.position = msg->pose.position;

        // we can set a desired yaw angle here if we need to sabrina, otherwise it will be IGNORED!!!
        target_msg.yaw = 0.0;

        // publish the final command for the flight controller
        target_pub_->publish(target_msg);
    }

} // namespace kestrel_control

// Run node
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<kestrel_control::ArduPilotTranslatorNode>());
  rclcpp::shutdown();
  return 0;
}

