#include <iostream>     // For std::cout, std::endl
#include <thread>       // For std::this_thread::sleep_for
#include <chrono>       // For std::chrono::milliseconds
#include <cmath>        // For math functions like std::sqrt and std::cos

// Function to simulate 4x4 sensor data representing a spherical object
int mockData() {
    const int rows = 4;
    const int cols = 4;

    const float radius = 2.0f;         // Simulated radius of the spherical object
    const float maxDepth = 10000.0f;   // Max sensor reading (e.g., closest point)

    // Infinite loop to simulate a live sensor feed
    while (true) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Calculate distance from the center of the 4x4 grid
                float dx = j - 1.5f;  // Horizontal distance from center column
                float dy = i - 1.5f;  // Vertical distance from center row
                float distance = std::sqrt(dx*dx + dy*dy);  // Euclidean distance

                // Simulate a spherical depth curve using cosine drop-off
                float value = maxDepth * std::cos((distance / radius) * (M_PI / 2.0f));
                
                // Clamp negative values to zero (sensor can't detect "negative distance")
                value = std::max(0.0f, value);

                // Output the value (comma-separated for each row)
                std::cout << value;
                if (j < cols - 1) std::cout << ",";
            }
            std::cout << std::endl;  // End of row
        }

        // Flush output so Python subprocess can immediately read new values
        std::flush(std::cout);

        // Simulate sensor frame rate (200ms between frames)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}

int main() {
    // Start mock sensor stream
    mockData();
}
