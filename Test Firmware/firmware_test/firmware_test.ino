// Arduino Nano — Hoverboard Diff-Drive + Odometry + MPU6050
// Protocol: "<v>,<w>\n"   v=m/s (fwd+), w=rad/s (CW+)
//           "r"           reset pose
// Output:   CSV  x,y,th,th_gyro,vL,vR,wL,wR,bat,temp

#include <SoftwareSerial.h>
#include <Wire.h>
#include <MPU6050_tockn.h>

SoftwareSerial HoverSerial(2, 3);   // RX=D2, TX=D3
MPU6050 mpu6050(Wire);

#define HOVER_SERIAL_BAUD   115200
#define SERIAL_BAUD         115200
#define START_FRAME         0xABCD
#define TIME_SEND           100

#define V_MAX_REAL   0.8f
#define W_MAX_REAL   3.0f
#define V_MAX_CMD    93
#define W_MAX_CMD    91

#define WHEEL_DIAMETER_M    0.165f
#define WHEEL_CIRCUM_M      (3.14159265f * WHEEL_DIAMETER_M)
#define WHEEL_SEPARATION_M  0.521f
#define SPEED_MEAS_TO_RPM   (0.0086f * 60.0f / WHEEL_CIRCUM_M)

#define WATCHDOG_MS         1000
#define POSE_PRINT_MS       100

// --- Fusion weights (0..1): how much to trust gyro vs wheels for theta ---
// 1.0 = full gyro, 0.0 = full wheel odometry
#define GYRO_WEIGHT         0.95f

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
float gyro_theta = 0.0f;      // integrated from MPU
float gyro_bias  = 0.0f;      // estimated while stationary
unsigned long lastOdomTime = 0;
bool odomInitialized = false;

String rxLine = "";
float  g_v = 0.0f, g_w = 0.0f;
bool   g_newCmd = false;

unsigned long iTimeSend = 0, iTimeWatch = 0, iTimePosePrint = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.print(F("x,y,th,th_gyro,vL,vR,wL,wR,bat,temp\n"));

  // MPU6050 init
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);   // keep robot still during this (~2 s)

  HoverSerial.begin(HOVER_SERIAL_BAUD);
  pinMode(LED_BUILTIN, OUTPUT);
}

void Send(int16_t steer, int16_t speed) {
  Command.start    = (uint16_t)START_FRAME;
  Command.steer    = steer;
  Command.speed    = speed;
  Command.checksum = (uint16_t)(Command.start ^ Command.steer ^ Command.speed);
  HoverSerial.write((uint8_t *)&Command, sizeof(Command));
}

void SendWheels(int16_t left, int16_t right) {
  left  = -left;
  right = -right;
  Send((right - left) / 2, (left + right) / 2);
}

void drive(float v, float w) {
  float v_cmd = (v / V_MAX_REAL) * V_MAX_CMD;
  float w_cmd = (w / W_MAX_REAL) * W_MAX_CMD;

  float mag = fmaxf(fabsf(v_cmd), fabsf(w_cmd));
  if (mag > (float)V_MAX_CMD) {
    float s = (float)V_MAX_CMD / mag;
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

void Receive() {
  if (HoverSerial.available()) {
    incomingByte  = HoverSerial.read();
    bufStartFrame = ((uint16_t)(incomingByte) << 8) | incomingBytePrev;
  } else {
    return;
  }

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
    }
    idx = 0;
  }
  incomingBytePrev = incomingByte;
}

void parseIncoming() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == 'r' || c == 'R') {
      odom_x = 0.0f; odom_y = 0.0f; odom_theta = 0.0f;
      gyro_theta = 0.0f;
      continue;
    }

    if (c == '\n' || c == '\r') {
      if (rxLine.length() > 0) {
        int comma = rxLine.indexOf(',');
        if (comma > 0) {
          g_v = rxLine.substring(0, comma).toFloat();
          g_w = rxLine.substring(comma + 1).toFloat();
          g_newCmd = true;
        }
        rxLine = "";
      }
    } else {
      rxLine += c;
      if (rxLine.length() > 32) rxLine = "";
    }
  }
}

float measToMps(int16_t meas, bool invert) {
  float rpm = (float)meas;
  if (invert) rpm = -rpm;
  rpm *= SPEED_MEAS_TO_RPM;
  return (rpm / 60.0f) * WHEEL_CIRCUM_M;
}

float measToRadS(int16_t meas, bool invert) {
  return measToMps(meas, invert) / (WHEEL_DIAMETER_M / 2.0f);
}

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

  // --- Wheel-based velocities ---
  float vL = measToMps(Feedback.speedL_meas, true);
  float vR = measToMps(Feedback.speedR_meas, false);

  float v            = (vL + vR) / 2.0f;
  float omega_wheels = -(vR - vL) / WHEEL_SEPARATION_M;   // CW+ -> CCW- for theta

  // --- Gyro-based angular velocity ---
  mpu6050.update();
  // getAngleZ returns degrees (tockn library). Convert to rad.
  // Sign: MPU Z is CW+ typically; adjust sign so that clockwise = negative theta
  // (our theta is CCW-positive). We keep gyro_theta in CCW-positive convention.
  float gz_dps = mpu6050.getGyroZ();                  // deg/s, raw
  float omega_gyro = -(gz_dps * (float)M_PI / 180.0f); // rad/s, CCW+

  // --- Fuse ---
  float omega = GYRO_WEIGHT * omega_gyro + (1.0f - GYRO_WEIGHT) * omega_wheels;

  // Integrate pose
  float theta_mid = odom_theta + omega * dt / 2.0f;
  odom_x     += v * cosf(theta_mid) * dt;
  odom_y     += v * sinf(theta_mid) * dt;
  odom_theta += omega * dt;

  while (odom_theta >  M_PI) odom_theta -= 2.0f * M_PI;
  while (odom_theta < -M_PI) odom_theta += 2.0f * M_PI;

  // Separate pure-gyro integration for comparison / debug
  gyro_theta += omega_gyro * dt;
  while (gyro_theta >  M_PI) gyro_theta -= 2.0f * M_PI;
  while (gyro_theta < -M_PI) gyro_theta += 2.0f * M_PI;
}

void printPose(unsigned long timeNow) {
  if (timeNow - iTimePosePrint < POSE_PRINT_MS) return;
  iTimePosePrint = timeNow;

  float vL = measToMps(Feedback.speedL_meas, true);
  float vR = measToMps(Feedback.speedR_meas, false);
  float wL = measToRadS(Feedback.speedL_meas, true);
  float wR = measToRadS(Feedback.speedR_meas, false);

  Serial.print(odom_x, 3);     Serial.print(',');
  Serial.print(odom_y, 3);     Serial.print(',');
  Serial.print(odom_theta, 3); Serial.print(',');
  Serial.print(gyro_theta, 3); Serial.print(',');
  Serial.print(vL, 3);         Serial.print(',');
  Serial.print(vR, 3);         Serial.print(',');
  Serial.print(wL, 3);         Serial.print(',');
  Serial.print(wR, 3);         Serial.print(',');
  Serial.print(Feedback.batVoltage); Serial.print(',');
  Serial.println(Feedback.boardTemp);
}

void loop() {
  unsigned long timeNow = millis();

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