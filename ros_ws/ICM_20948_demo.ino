#include <Wire.h>
#include "SparkFun_ICM-20948_ArduinoLibrary.h"
#include <math.h>

// Create an ICM-20948 object for I2C
ICM_20948_I2C myICM;

void setup() {
  Serial.begin(115200);   // 
  Wire.begin();           // Initialize I2C
  Wire.setClock(400000);  // 

  // Initialize the ICM-20948 sensor
  if (myICM.begin(Wire) != ICM_20948_Stat_Ok) {
    Serial.println("ICM-20948 initialization failed. Check connections or I2C address!");
    while (1);
  }
  Serial.println("ICM-20948 initialized successfully!");

  // Set custom ranges for accelerometer or gyroscope
  // myICM.setFullScaleAccelRange(ICM_20948_ACCEL_RANGE_2G);   // Possible values: 2G, 4G, 8G, 16G
  // myICM.setFullScaleGyroRange(ICM_20948_GYRO_RANGE_250DPS); // Possible values: 250, 500, 1000, 2000 DPS
}

void loop() {
  if (myICM.dataReady()) {
    myICM.getAGMT();

    // Read accelerometer data (in g)
    float ax = myICM.accX();
    float ay = myICM.accY();
    float az = myICM.accZ();

    // Read gyroscope data (in degrees per second)
    float gx = myICM.gyrX();
    float gy = myICM.gyrY();
    float gz = myICM.gyrZ();

    // Calculate Pitch and Roll (in degrees)
    // Adjust signs or formulas if needed based on your sensor orientation.
    float pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / M_PI;
    float roll  = atan2(ay, az) * 180.0 / M_PI;
    static float yaw = 90.0;

    // Print data in CSV format: AccelX, AccelY, AccelZ, Pitch, Roll, Yaw
    Serial.print(ax, 2);
    Serial.print(", ");
    Serial.print(ay, 2);
    Serial.print(", ");
    Serial.print(az, 2);
    Serial.print(", ");
    Serial.print(pitch, 2);
    Serial.print(", ");
    Serial.print(roll, 2);
    Serial.print(", ");
    Serial.println(yaw, 2);
  }

  // Delay 
  delay(500);
}
