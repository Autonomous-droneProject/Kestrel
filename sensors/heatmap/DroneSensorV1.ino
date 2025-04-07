#include <Wire.h>
#include <VL53L1X.h>

VL53L1X sensor;
int LEDPIN = 17;

void setup() {
  Serial.begin(9600);
  delay(100); // Ensure Serial stabilizes
  Wire.begin(21, 22); // Use default ESP32 I2C pins or specify custom ones
  Wire.setClock(400000); // 400 kHz I2C

  pinMode(LEDPIN, OUTPUT);

  sensor.setTimeout(500);
  if (!sensor.init()) {
    Serial.println("Failed to detect and initialize VL53L1X sensor!");
    while (1); // Freeze execution
  }

  sensor.setROISize(16, 16);

  sensor.setDistanceMode(VL53L1X::Long);
  sensor.setMeasurementTimingBudget(50000);
  sensor.startContinuous(50);

  Serial.println("Sensor initialized and running.");
}

void loop() {
  int distance = sensor.read();
  if (sensor.timeoutOccurred()) {
    Serial.println("Sensor timeout!");
  } else {
    distance -= 20;
    if (distance < 1000)
      digitalWrite(LEDPIN, HIGH);
    else
      digitalWrite(LEDPIN, LOW);
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" mm");
  }
  delay(100);
}





