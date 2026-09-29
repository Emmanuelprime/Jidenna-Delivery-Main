// Arduino Nano — Hoverboard Diff-Drive + Odometry + MPU6050
// Input:  "<v>,<w>\n"  v=m/s (fwd+), w=rad/s (CCW+)   |  "r" reset pose
// Output: CSV  x,y,th,vL,vR,wL,wR,bat,temp,fb_age,wd,or,imu_yaw
//
// CONVENTION (measured on hardware, 2026-09-29):
//   w > 0  -> left wheel backward, right wheel forward -> robot rotates CCW
//   w < 0  -> left wheel forward,  right wheel backward -> robot rotates CW
//   th     -> decreases for w > 0, increases for w < 0
//   NOTE: this is opposite to the original spec ("CW+"). The behavior above
//   is what the hardware actually does. Treat w as CCW-positive on the Pi side.
//
//   v > 0  -> both wheels forward -> x increases
//   v < 0  -> both wheels backward -> x decreases
//
//   imu_yaw = MPU6050 integrated yaw in degrees. Independent of wheel odom.
//             Will drift over time. Useful for detecting wheel slip / lift.

#include <SoftwareSerial.h>
#include <Wire.h>
#include <MPU6050_tockn.h>
#include <stdlib.h>   // strtod, strchr
#include <string.h>   // memcpy

SoftwareSerial HoverSerial(2, 3);   // RX=D2, TX=D3

MPU6050 mpu6050(Wire);

#define HOVER_SERIAL_BAUD   115200
#define SERIAL_BAUD         115200
#define START_FRAME         0xABCD
#define TIME_SEND           100
#define POSE_PRINT_MS       100
#define WATCHDOG_MS         1000
#define FEEDBACK_STALE_MS   500
#define IMU_UPDATE_MS       50      // 20 Hz — enough for telemetry, easy on loop

#define V_MAX_REAL   0.8f
#define W_MAX_REAL   3.0f
#define V_MAX_CMD    93
#define W_MAX_CMD    91

#define WHEEL_DIAMETER_M    0.165f
#define WHEEL_CIRCUM_M      (3.14159265f * WHEEL_DIAMETER_M)
#define WHEEL_SEPARATION_M  0.521f
#define SPEED_MEAS_TO_RPM   (0.0086f * 60.0f / WHEEL_CIRCUM_M)

// Hoverboard feedback unit conversions (raw counts -> engineering units)
#define HOVER_BAT_UNIT_V    0.01f    // raw * 0.01 = volts
#define HOVER_TEMP_UNIT_C   0.1f     // raw * 0.1  = degC

typedef struct {
  uint16_t start;
  int16_t  steer;
  int16_t  speed;
  uint16_t checksum;
} SerialCommand;
SerialCommand Command;

typedef struct {
  uint16_t start;
  int16_t  cmd1;
  int16_t  cmd2;
  int16_t  speedR_meas;
  int16_t  speedL_meas;
  int16_t  batVoltage;
  int16_t  boardTemp;
  uint16_t cmdLed;
  uint16_t checksum;
} SerialFeedback;
SerialFeedback Feedback, NewFeedback;

uint8_t  idx = 0;
uint16_t bufStartFrame;
byte    *p;
byte     incomingByte, incomingBytePrev;

float odom_x = 0.0f, odom_y = 0.0f, odom_theta = 0.0f;
unsigned long lastOdomTime = 0;
bool odomInitialized = false;

// Fixed char buffer instead of String (no heap fragmentation)
char    rxBuf[40];
uint8_t rxLen = 0;

float  g_v = 0.0f, g_w = 0.0f;
bool   g_newCmd = false;

unsigned long iTimeSend = 0, iTimeWatch = 0, iTimePosePrint = 0;
unsigned long lastFeedbackMs = 0;
bool          haveFeedback = false;
uint32_t      odomResetCount = 0;

// IMU rate limiting
unsigned long lastImuUpdate = 0;

// ---- helpers ---------------------------------------------------------------

static inline bool isFiniteF(float v) {
  // NaN-safe, Inf-safe, no <math.h> isinf needed on AVR
  return (v == v) && ((v - v) == 0.0f);
}

// ---- setup -----------------------------------------------------------------

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println(F("x,y,th,vL,vR,wL,wR,bat,temp,fb_age,wd,or,imu_yaw"));

  HoverSerial.begin(HOVER_SERIAL_BAUD);
  pinMode(LED_BUILTIN, OUTPUT);

  // MPU6050 on I2C (A4=SDA, A5=SCL). SoftwareSerial uses D2/D3, no conflict.
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);   // ~3s, prints to Serial — fine at boot
}

// ---- hoverboard TX ---------------------------------------------------------

void Send(int16_t steer, int16_t speed) {
  Command.start    = (uint16_t)START_FRAME;
  Command.steer    = steer;
  Command.speed    = speed;
  Command.checksum = (uint16_t)(Command.start ^ Command.steer ^ Command.speed);
  HoverSerial.write((uint8_t *)&Command, sizeof(Command));
}

// Unchanged from working baseline
void SendWheels(int16_t left, int16_t right) {
  left  = -left;
  right = -right;
  Send((right - left) / 2, (left + right) / 2);
}

void drive(float v, float w) {
  float v_cmd = (v / V_MAX_REAL) * V_MAX_CMD;
  float w_cmd = (w / W_MAX_REAL) * W_MAX_CMD;

  // L1-norm saturation: preserves curvature and prevents one wheel
  // exceeding V_MAX_CMD (the old max(|v|,|w|) norm allowed vL=186).
  float l1    = fabsf(v_cmd) + fabsf(w_cmd);
  float limit = (float)V_MAX_CMD;
  if (l1 > limit) {
    float s = limit / l1;
    v_cmd *= s;
    w_cmd *= s;
  }

  int16_t vL = (int16_t)(v_cmd + w_cmd);
  int16_t vR = (int16_t)(v_cmd - w_cmd);
  if (vL >  1000) vL =  1000;
  if (vL < -1000) vL = -1000;
  if (vR >  1000) vR =  1000;
  if (vR < -1000) vR = -1000;

  SendWheels(vL, vR);
}

// ---- hoverboard RX ---------------------------------------------------------

// Unchanged from working baseline (start-frame detection matches your hardware)
void Receive() {
  while (HoverSerial.available()) {
    incomingByte  = HoverSerial.read();
    bufStartFrame = ((uint16_t)(incomingByte) << 8) | incomingBytePrev;

    if (bufStartFrame == START_FRAME) {
      p    = (byte *)&NewFeedback;
      *p++ = incomingBytePrev;
      *p++ = incomingByte;
      idx  = 2;
    } else if (idx >= 2 && idx < sizeof(SerialFeedback)) {
      *p++ = incomingByte;
      idx++;
    }

    if (idx == sizeof(SerialFeedback)) {
      uint16_t checksum = (uint16_t)(NewFeedback.start ^ NewFeedback.cmd1 ^ NewFeedback.cmd2
                          ^ NewFeedback.speedR_meas ^ NewFeedback.speedL_meas
                          ^ NewFeedback.batVoltage  ^ NewFeedback.boardTemp
                          ^ NewFeedback.cmdLed);
      if (NewFeedback.start == START_FRAME && checksum == NewFeedback.checksum) {
        memcpy(&Feedback, &NewFeedback, sizeof(SerialFeedback));
        lastFeedbackMs = millis();
        haveFeedback   = true;
      }
      idx = 0;
    }
    incomingBytePrev = incomingByte;
  }
}

// ---- host command parsing --------------------------------------------------

// Strict validation; rejects bad input rather than silently producing 0,0.
void parseIncoming() {
  while (Serial.available()) {
    char c = (char)Serial.read();

    if (c == 'r' || c == 'R') {
      odom_x = odom_y = odom_theta = 0.0f;
      odomResetCount++;
      rxLen = 0;
      continue;
    }

    if (c == '\n' || c == '\r') {
      if (rxLen == 0) continue;
      rxBuf[rxLen] = '\0';

      char* comma = strchr(rxBuf, ',');
      if (comma == nullptr) { rxLen = 0; continue; }
      if (strchr(comma + 1, ',') != nullptr) { rxLen = 0; continue; }
      if (comma == rxBuf || *(comma + 1) == '\0') { rxLen = 0; continue; }

      *comma = '\0';

      // strtof not on AVR; strtod is (double==float on ATmega328P)
      char* endV = nullptr;
      char* endW = nullptr;
      float v = (float)strtod(rxBuf, &endV);
      float w = (float)strtod(comma + 1, &endW);

      if (endV == rxBuf || *endV != '\0' ||
          endW == comma + 1 || *endW != '\0') { rxLen = 0; continue; }

      if (!isFiniteF(v) || !isFiniteF(w)) { rxLen = 0; continue; }

      if (v < -V_MAX_REAL || v > V_MAX_REAL ||
          w < -W_MAX_REAL || w > W_MAX_REAL) { rxLen = 0; continue; }

      g_v = v;
      g_w = w;
      g_newCmd = true;
      rxLen = 0;
    } else {
      if (rxLen < sizeof(rxBuf) - 1) {
        rxBuf[rxLen++] = c;
      } else {
        rxLen = 0;   // overflow; resync on next newline
      }
    }
  }
}

// ---- feedback -> physical units --------------------------------------------

float measToMps(int16_t meas, bool invert) {
  float rpm = (float)meas;
  if (invert) rpm = -rpm;
  return (rpm * SPEED_MEAS_TO_RPM / 60.0f) * WHEEL_CIRCUM_M;
}

float measToRadS(int16_t meas, bool invert) {
  return measToMps(meas, invert) / (WHEEL_DIAMETER_M / 2.0f);
}

// ---- odometry --------------------------------------------------------------

void updateOdometry() {
  unsigned long now = millis();

  if (!odomInitialized) {
    lastOdomTime = now;
    odomInitialized = true;
    return;
  }

  float dt = (now - lastOdomTime) / 1000.0f;
  lastOdomTime = now;
  if (dt <= 0.0f || dt > 0.5f) return;

  // Don't integrate stale feedback (prevents frozen values from drifting)
  if (!haveFeedback || (now - lastFeedbackMs) > FEEDBACK_STALE_MS) return;

  float vL = measToMps(Feedback.speedL_meas, true);
  float vR = measToMps(Feedback.speedR_meas, false);

  float v     = (vL + vR) / 2.0f;
  float omega = -(vR - vL) / WHEEL_SEPARATION_M;

  float theta_mid = odom_theta + omega * dt / 2.0f;
  odom_x     += v * cosf(theta_mid) * dt;
  odom_y     += v * sinf(theta_mid) * dt;
  odom_theta += omega * dt;

  while (odom_theta >  M_PI) odom_theta -= 2.0f * M_PI;
  while (odom_theta < -M_PI) odom_theta += 2.0f * M_PI;
}

// ---- telemetry -------------------------------------------------------------

void printPose(unsigned long timeNow) {
  if (timeNow - iTimePosePrint < POSE_PRINT_MS) return;
  iTimePosePrint = timeNow;

  // Use a fresh millis() for fb_age so we don't underflow if Receive()
  // updated lastFeedbackMs after loop() captured timeNow.
  uint32_t nowMs = millis();
  uint32_t fbAge;
  if (!haveFeedback) {
    fbAge = 9999;
  } else {
    fbAge = (nowMs >= lastFeedbackMs) ? (nowMs - lastFeedbackMs) : 0;
  }
  if (fbAge > 9999) fbAge = 9999;

  bool wd = (iTimeWatch != 0 && (timeNow - iTimeWatch) > WATCHDOG_MS);

  Serial.print(odom_x, 3);     Serial.print(',');
  Serial.print(odom_y, 3);     Serial.print(',');
  Serial.print(odom_theta, 3); Serial.print(',');
  Serial.print(measToMps(Feedback.speedL_meas, true), 3);  Serial.print(',');
  Serial.print(measToMps(Feedback.speedR_meas, false), 3); Serial.print(',');
  Serial.print(measToRadS(Feedback.speedL_meas, true), 3); Serial.print(',');
  Serial.print(measToRadS(Feedback.speedR_meas, false), 3);Serial.print(',');

  // Convert raw counts to engineering units
  Serial.print((float)Feedback.batVoltage * HOVER_BAT_UNIT_V, 2);  Serial.print(',');   // V
  Serial.print((float)Feedback.boardTemp  * HOVER_TEMP_UNIT_C, 1); Serial.print(',');   // C

  Serial.print(fbAge);      Serial.print(',');
  Serial.print(wd ? 1 : 0); Serial.print(',');
  Serial.print(odomResetCount); Serial.print(',');
  Serial.println(mpu6050.getAngleZ(), 2);   // yaw in degrees
}

// ---- loop ------------------------------------------------------------------

void loop() {
  unsigned long timeNow = millis();

  // Rate-limit IMU update so Wire's blocking I2C read doesn't starve
  // SoftwareSerial at 115200 baud. 20 Hz is plenty for telemetry.
  if (timeNow - lastImuUpdate >= IMU_UPDATE_MS) {
    lastImuUpdate = timeNow;
    mpu6050.update();
  }

  parseIncoming();
  Receive();
  updateOdometry();

  if (iTimeSend <= timeNow) {
    iTimeSend = timeNow + TIME_SEND;

    if (g_newCmd) {
      iTimeWatch = timeNow;
      g_newCmd  = false;
    }
    if (iTimeWatch != 0 && (timeNow - iTimeWatch) > WATCHDOG_MS) {
      g_v = 0.0f;
      g_w = 0.0f;
    }

    drive(g_v, g_w);
  }

  printPose(timeNow);
  digitalWrite(LED_BUILTIN, (timeNow % 2000) < 1000);
}