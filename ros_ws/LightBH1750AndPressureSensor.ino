#include <Wire.h>
#include <BH1750.h>

BH1750 lightMeter;          
int sensorPin = A0;          
int signal;                  

unsigned long previousMillis = 0;       
const unsigned long interval = 500;      

void setup() {
  Serial.begin(9600);
  Wire.begin();
  lightMeter.begin();        
}

void loop() {
  unsigned long currentMillis = millis();
  

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    

    signal = analogRead(sensorPin);
    

    float lux = lightMeter.readLightLevel();

    Serial.print(currentMillis);
    Serial.print(" ");
    Serial.print(signal);
    Serial.print(" ");
    Serial.print(lux);
    Serial.println(" lux");
  }
}
