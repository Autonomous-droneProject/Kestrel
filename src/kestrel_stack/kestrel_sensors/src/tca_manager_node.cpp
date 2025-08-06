#include "rclcpp/rclcpp.hpp"
#include "kestrel_sensors/srv/select_channel.hpp" // Auto-generated from .srv file
#include "tca9548a/tca9548a.hpp" // Your driver header

#include <memory>

class TcaManagerNode : public rclcpp::Node {
public:
    TcaManagerNode() : Node("tca_manager_node") {
        // Declare parameters for I2C bus and address
        this->declare_parameter<std::string>("i2c_bus", "/dev/i2c-1");
        this->declare_parameter<int>("i2c_address", 0x70);

        std::string i2c_bus = this->get_parameter("i2c_bus").as_string();
        uint8_t i2c_address = this->get_parameter("i2c_address").as_int();

        // Initialize the driver
        tca_driver_ = std::make_unique<Tca9548a>(i2c_bus, i2c_address);
        if (!tca_driver_->open_bus()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open I2C bus for TCA9548A.");
            rclcpp::shutdown();
            return;
        }

        // Create the service
        service_ = this->create_service<kestrel_sensors::srv::SelectChannel>(
            "select_channel",
            std::bind(&TcaManagerNode::handle_select_channel, this, std::placeholders::_1, std::placeholders::_2)
        );

        RCLCPP_INFO(this->get_logger(), "TCA9548A Manager is ready. Awaiting channel selection requests.");
    }

    ~TcaManagerNode() {
        tca_driver_->close_bus();
    }

private:
    void handle_select_channel(
        const std::shared_ptr<kestrel_sensors::srv::SelectChannel::Request> request,
        std::shared_ptr<kestrel_sensors::srv::SelectChannel::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Request received to select channel %d", request->channel);
        
        if (request->channel > 7) {
            response->success = false;
            response->message = "Channel must be between 0 and 7.";
            RCLCPP_ERROR(this->get_logger(), response->message.c_str());
            return;
        }

        if (tca_driver_->select_channel(request->channel)) {
            response->success = true;
            response->message = "Successfully selected channel " + std::to_string(request->channel);
        } else {
            response->success = false;
            response->message = "Failed to select channel " + std::to_string(request->channel);
            RCLCPP_ERROR(this->get_logger(), response->message.c_str());
        }
    }

    std::unique_ptr<Tca9548a> tca_driver_;
    rclcpp::Service<kestrel_sensors::srv::SelectChannel>::SharedPtr service_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TcaManagerNode>());
    rclcpp::shutdown();
    return 0;
}