const int sensorPin = 4;          // GPIO 4 - Entrada analogica MyoWare
const int SAMPLE_INTERVAL = 1000; // 1000 us -> fs = 1000 Hz
const int BUFFER_SIZE = 50;

int rawBuffer[BUFFER_SIZE];
int bufferIndex = 0;
unsigned long lastMicros = 0;

void setup() {
  Serial.begin(115200);
  
  unsigned long startTime = millis();
  while (!Serial && (millis() - startTime < 3000)) {
    // Espera de conexion serial
  }
  
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); 
}

void loop() {
  unsigned long currentMicros = micros();

  if (currentMicros - lastMicros >= SAMPLE_INTERVAL) {
    lastMicros += SAMPLE_INTERVAL;

    rawBuffer[bufferIndex] = analogRead(sensorPin);
    bufferIndex++;

    if (bufferIndex >= BUFFER_SIZE) {
      for (int i = 0; i < BUFFER_SIZE; i++) {
        Serial.println(rawBuffer[i]);
      }
      bufferIndex = 0;
    }
  }
}
