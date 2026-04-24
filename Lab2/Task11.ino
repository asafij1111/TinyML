#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

// ===== Previous values =====
float prevTemp = 0;
float prevHumidity = 0;
float prevMag = 0;
int prevClear = 0;

// ===== Cooldown =====
unsigned long lastEventTime = 0;
const unsigned long COOLDOWN_MS = 2000;

// ===== Thresholds =====
const float HUMID_THRESHOLD = 10.0;
const float TEMP_THRESHOLD = 1.0;
const float MAG_THRESHOLD = 5000.0;
const int LIGHT_THRESHOLD = 2000;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!HS300x.begin()) {
    Serial.println("HS300x failed!");
    while (1);
  }

  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }

  if (!APDS.begin()) {
    Serial.println("APDS failed!");
    while (1);
  }

  Serial.println("Task 11 system ready");
}

void loop() {

  // ===== 1. READ SENSORS =====
  float temp = HS300x.readTemperature();
  float humidity = HS300x.readHumidity();

  float mx, my, mz;
  float mag = 0;
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    mag = abs(mx) + abs(my) + abs(mz);
  }

  int r, g, b, clear = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clear);
  }

  // ===== 2. COMPUTE DELTAS =====
  float dHumidity = humidity - prevHumidity;
  float dTemp = temp - prevTemp;
  float dMag = abs(mag - prevMag);
  int dLight = abs(clear - prevClear);

  // ===== 3. EVENT FLAGS =====
  bool humid_jump = (dHumidity > HUMID_THRESHOLD);
  bool temp_rise = (dTemp > TEMP_THRESHOLD);
  bool mag_shift = (dMag > MAG_THRESHOLD);
  bool light_change = (dLight > LIGHT_THRESHOLD);

  // ===== 4. EVENT CLASSIFICATION =====
  String label = "BASELINE_NORMAL";

  if (humid_jump || temp_rise) {
    label = "BREATH_OR_WARM_AIR_EVENT";
  }
  else if (mag_shift) {
    label = "MAGNETIC_DISTURBANCE_EVENT";
  }
  else if (light_change) {
    label = "LIGHT_OR_COLOR_CHANGE_EVENT";
  }

  // ===== 5. SERIAL OUTPUT =====
  Serial.println("-----");

  Serial.print("rh=");
  Serial.print(humidity);
  Serial.print(" temp=");
  Serial.print(temp);
  Serial.print(" mag=");
  Serial.print(mag);
  Serial.print(" clear=");
  Serial.println(clear);

  Serial.print("humid_jump=");
  Serial.print(humid_jump);
  Serial.print(" temp_rise=");
  Serial.print(temp_rise);
  Serial.print(" mag_shift=");
  Serial.print(mag_shift);
  Serial.print(" light_or_color_change=");
  Serial.println(light_change);

  // ===== 6. COOLDOWN LOGIC =====
  if (label != "BASELINE_NORMAL") {
    if (millis() - lastEventTime > COOLDOWN_MS) {
      Serial.print("FINAL_LABEL=");
      Serial.println(label);
      lastEventTime = millis();
    }
  } else {
    Serial.print("FINAL_LABEL=");
    Serial.println("BASELINE_NORMAL");
  }

  // ===== 7. UPDATE PREVIOUS VALUES =====
  prevHumidity = humidity;
  prevTemp = temp;
  prevMag = mag;
  prevClear = clear;

  delay(1000);
}
