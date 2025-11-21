#include "kestrel_control/ardupilot_translator_node.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "kestrel_msgs/srv/set_flight_mode.hpp"
#include "kestrel_msgs/msg/flight_status.hpp"

namespace kestrel_control
{

    ArduPilotTranslatorNode::ArduPilotTranslatorNode(const rclcpp::NodeOptions &options)
        : Node("ardupilot_translator_node", options)
    {
        initialize();
    }

    void ArduPilotTranslatorNode::handle_emergency_stop(
    const std::shared_ptr<kestrel_msgs::srv::TriggerEmergencyStop::Request> request,
    std::shared_ptr<kestrel_msgs::srv::TriggerEmergencyStop::Response> response)
    {
        RCLCPP_FATAL(this->get_logger(), "EMERGENCY STOP: %s", request->reason.c_str());
        if (set_mode_client_->wait_for_service(std::chrono::seconds(1))) {
            auto mode_request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
            mode_request->custom_mode = "LAND";  // ArduPilot LAND mode
            set_mode_client_->async_send_request(mode_request);
        }
        response->acknowledged = true;
        response->message = "Emergency stop executed: " + request->reason;
    }

    void ArduPilotTranslatorNode::handle_set_flight_mode(
    const std::shared_ptr<kestrel_msgs::srv::SetFlightMode::Request> request,
    std::shared_ptr<kestrel_msgs::srv::SetFlightMode::Response> response)
    {
        if (set_mode_client_->wait_for_service(std::chrono::seconds(1))) {
            auto mode_request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
            
            // mapping kestrel flight modes to ArduPilot modes
            switch(request->flight_mode) {
                case kestrel_msgs::msg::FlightStatus::MODE_MANUAL:
                    mode_request->custom_mode = "STABILIZE";
                    break;
                case kestrel_msgs::msg::FlightStatus::MODE_AUTO:
                    mode_request->custom_mode = "GUIDED";
                    break;
                case kestrel_msgs::msg::FlightStatus::MODE_LANDING:
                    mode_request->custom_mode = "LAND";
                    break;
                default:
                    mode_request->custom_mode = "LOITER";
            }
            
            set_mode_client_->async_send_request(mode_request);
            response->success = true;
            response->message = "Flight mode change requested";
        } else {
            response->success = false;
            response->message = "MAVROS set_mode service not available";
        }
    }

    void ArduPilotTranslatorNode::initialize()
    {
        // setup publisher and subscriber
        target_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>("mavros/setpoint_raw/local", 10);
        waypoint_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "planning/waypoint", 10, std::bind(&ArduPilotTranslatorNode::waypoint_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "ArduPilotTranslatorNode initialized.");

        // creating the emergency stop service
        emergency_stop_service_ = this->create_service<kestrel_msgs::srv::TriggerEmergencyStop>(
            "services/trigger_emergency_stop",
            std::bind(&ArduPilotTranslatorNode::handle_emergency_stop, this, 
                    std::placeholders::_1, std::placeholders::_2));

        // creating MAVROS client for mode changes
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");

        set_flight_mode_service_ = this->create_service<kestrel_msgs::srv::SetFlightMode>(
        "services/set_flight_mode",
        std::bind(&ArduPilotTranslatorNode::handle_set_flight_mode, this,
                std::placeholders::_1, std::placeholders::_2));
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

