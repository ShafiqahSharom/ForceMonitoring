#include <HX711_ADC.h>
#if defined(ESP8266)|| defined(ESP32) || defined(AVR)
#include <EEPROM.h>
#endif

// Pins:
const int HX711_dout = 4; // MCU > HX711 dout pin
const int HX711_sck = 5;  // MCU > HX711 sck pin

// HX711 constructor:
HX711_ADC LoadCell(HX711_dout, HX711_sck);
const int serialPrintInterval = 0; // Serial print interval in milliseconds

unsigned long t = 0;
float tareValue = 0.0; // Variable to store the tare value

void setup() {
  Serial.begin(115200); 
  Serial.println();
  Serial.println("Starting...");

  LoadCell.begin();
  unsigned long stabilizingtime = 2000;
  boolean _tare = true;
  LoadCell.start(stabilizingtime, _tare);
  
  // Additional delay before setting tare value
  delay(5000); // Adjust the delay time as needed
  
  if (LoadCell.getTareTimeoutFlag() || LoadCell.getSignalTimeoutFlag()) {
    Serial.println("Timeout, check MCU > HX711 wiring and pin designations");
    while (1);
  } else {
    // Set tare value after additional delay
    tareValue = LoadCell.getData();
    Serial.println(tareValue);
    Serial.println("Tare value set. Startup is complete");
  }
}

void loop() {
  static boolean newDataReady = false;
  

  // Check for new data/start next conversion:
  if (LoadCell.update()) newDataReady = true;

  // Get smoothed value from the dataset:
  if (newDataReady) {
    if (millis() > t + serialPrintInterval) {
      float relativeWeight = LoadCell.getData() - tareValue;
      Serial.print("Relative weight: ");
      Serial.println(relativeWeight);
      newDataReady = false;
    }
  }

  // Receive command from serial terminal
  if (Serial.available() > 0) {
    char inByte = Serial.read();
    if (inByte == 't') {
      // Set tare value manually if needed
      tareValue = LoadCell.getData();
      Serial.println("Tare value set manually");
    }
  }
}
