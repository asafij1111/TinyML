#include <PDM.h>
#include <Arduino_APDS9960.h>
#include <Arduino_BMI270_BMM150.h>

// ===== Microphone =====
short sampleBuffer[256];
volatile int samplesRead = 0;
int micLevel = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

// ===== Global sensor values =====
int clearVal = 0;
float ax, ay, az;
int prox = 0;

// ===== Thresholds (TUNE THESE IF NEEDED) =====
const int SOUND_THRESHOLD = 2000;
const int LIGHT_THRESHOLD = 50;
const float MOTION_THRESHOLD = 1;
const int PROX_THRESHOLD = 100;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Mic failed!");
    while (1);
  }
  PDM.setGain(40);

  // Light + proximity
  if (!APDS.begin()) {
    Serial.println("APDS failed!");
    while (1);
  }

  // IMU
  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }

  Serial.println("System ready");
}

void loop() {

  // ===== 1. MICROPHONE =====
  if (samplesRead) {
    long sum = 0;
    for (int i = 0; i < samplesRead; i++) {
      sum += abs(sampleBuffer[i]);
    }
    micLevel = sum / samplesRead;
    samplesRead = 0;
  }

  // ===== 2. LIGHT =====
  int r, g, b;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clearVal);
  }

  // ===== 3. PROXIMITY =====
  if (APDS.proximityAvailable()) {
    prox = APDS.readProximity();
  }

  // ===== 4. MOTION =====
  float motion = 0;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    motion = abs(ax) + abs(ay) + abs(az - 1.0); // deviation from gravity
  }

  // ===== Binary Decisions =====
  bool sound = (micLevel > SOUND_THRESHOLD);
  bool dark = (clearVal < LIGHT_THRESHOLD);
  bool moving = (motion > MOTION_THRESHOLD);
  bool near = (prox < PROX_THRESHOLD);

  // ===== Classification =====
  String label = "UNKNOWN";

  if (!sound && !dark && !moving && !near) {
    label = "QUIET_BRIGHT_STEADY_FAR";
  }
  else if (sound && !dark && !moving && !near) {
    label = "NOISY_BRIGHT_STEADY_FAR";
  }
  else if (!sound && dark && !moving && near) {
    label = "QUIET_DARK_STEADY_NEAR";
  }
  else if (sound && !dark && moving && near) {
    label = "NOISY_BRIGHT_MOVING_NEAR";
  }

  // ===== Required Serial Output =====
  Serial.println("-----");

  Serial.print("mic=");
  Serial.print(micLevel);
  Serial.print(", clear=");
  Serial.print(clearVal);
  Serial.print(", motion=");
  Serial.print(motion);
  Serial.print(", prox=");
  Serial.println(prox);

  Serial.print("sound=");
  Serial.print(sound);
  Serial.print(", dark=");
  Serial.print(dark);
  Serial.print(", moving=");
  Serial.print(moving);
  Serial.print(", near=");
  Serial.println(near);

  Serial.print("state, ");
  Serial.println(label);

  delay(1000);
}