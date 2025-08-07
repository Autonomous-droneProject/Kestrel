def generate_launch_description():

    # this node is responsible for all sensors on the i2c bus lol
    i2c_driver_node = Node(
        package='kestrel_perception',
        executable='i2c_driver_node',
        name='i2c_driver_node',
        output='screen'
    )

    # this node is responsible for all sensors on the UART bus
    uart_driver_node = Node(
        package='kestrel_perception',
        executable='uart_driver_node',
        name='uart_driver_node',
        output='screen'
    )

    return LaunchDescription([
        i2c_driver_node,
        uart_driver_node
        # i mean you can add more driver nodes here if you want aldem lol
    ])