#include <Wire.h>
#include <VL53L1X.h>

VL53L1X sensor;

int distance[16];
uint8_t spadCenters[16]= {10, 42, 74, 106, 14, 46, 78, 110,245, 213, 181, 149, 241, 209, 177, 145};

void setup() {
  Serial.begin(9600);
  delay(100);
  Wire.begin(21, 22);
  Wire.setClock(400000);

  sensor.setTimeout(500);
  if (!sensor.init()) {
    Serial.println("Failed to detect and initialize VL53L1X sensor!");
    while (1); // Freeze execution
  }

  sensor.setDistanceMode(VL53L1X::Long);
  sensor.setMeasurementTimingBudget(50000);
  sensor.startContinuous(50);
  sensor.setROISize(4,4);

  Serial.println("Sensor initialized and running.");
}

void loop() {
  int k = 0;
  if (sensor.timeoutOccurred())
    Serial.println("Sensor Timeout!");
  else {
    for (int i = 0; i < 16; i++)
    {
      distance[i] = readDistance(spadCenters[i]);
      Serial.print(distance[i]);
      Serial.print(",");
    }
    Serial.println();
  }
}

int readDistance(int spad)
{
  sensor.setROICenter(spad);
  delay(10);
  return sensor.read();
}

