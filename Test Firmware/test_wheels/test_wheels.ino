// *******************************************************************
//  Arduino Nano — Hoverboard Forward Drive + Speed Readout
//  Drives both wheels forward at SPEED, and prints measured
//  wheel speeds in m/s (and RPM) to the USB serial monitor.
// *******************************************************************

#include <SoftwareSerial.h>
SoftwareSerial HoverSerial(2, 3);   // RX, TX

// ============================================================
//  CHANGE THIS to set the wheel speed.
//  Range: 0 (stop) to ~1000 (max). Start low, like 100.
// ============================================================
#define SPEED   100

// ============================================================
//  Wheel geometry — used to convert raw readings to m/s.
// ============================================================
#define WHEEL_DIAMETER_M    0.165f
#define WHEEL_CIRCUM_M      (3.14159265f * WHEEL_DIAMETER_M)   // 0.5184 m

// ============================================================
//  Calibration: raw_meas * SPEED_MEAS_TO_RPM = wheel RPM.
//  Start with 1.0, then correct once you've compared to a
//  tape-measure distance test.
// ============================================================
#define SPEED_MEAS_TO_RPM   1.0f

// ############################################################

#define HOVER_SERIAL_BAUD   115200
#define SERIAL_BAUD         115200
#define START_FRAME         0xABCD
#define TIME_SEND           100     // [ms] send interval
#define PRINT_MS            500     // print speeds every 500 ms

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
SerialFeedback Feedback;
SerialFeedback NewFeedback;

// RX parser state
uint8_t  idx = 0;
uint16_t bufStartFrame;
byte    *p;
byte     incomingByte;
byte     incomingBytePrev;

// Timers
unsigned long iTimeSend  = 0;
unsigned long iTimePrint = 0;

// ########################## SETUP ##########################
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println(F("Forward Drive + Speed Readout"));
  Serial.print(F("Wheel circumference = ")); Serial.print(WHEEL_CIRCUM_M, 4); Serial.println(F(" m"));
  Serial.print(F("Commanded speed     = ")); Serial.println(SPEED);
  Serial.println(F("Format:  vL=<m/s>  vR=<m/s>  bat=<V>  temp=<C>"));

  HoverSerial.begin(HOVER_SERIAL_BAUD);
  pinMode(LED_BUILTIN, OUTPUT);
}

// ########################## SEND ##########################
void Send(int16_t steer, int16_t speed) {
  Command.start    = (uint16_t)START_FRAME;
  Command.steer    = steer;
  Command.speed    = speed;
  Command.checksum = (uint16_t)(Command.start ^ Command.steer ^ Command.speed);
  HoverSerial.write((uint8_t *)&Command, sizeof(Command));
}

// ########################## RECEIVE ##########################
void Receive() {
  if (HoverSerial.available()) {
    incomingByte    = HoverSerial.read();
    bufStartFrame   = ((uint16_t)(incomingByte) << 8) | incomingBytePrev;
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
    uint16_t checksum;
    checksum = (uint16_t)(NewFeedback.start ^ NewFeedback.cmd1 ^ NewFeedback.cmd2
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

// ########################## SPEED CONVERSION ##########################
// Raw hoverboard reading -> wheel linear speed [m/s]
//   invert=true for LEFT wheel  (its feedback sign is mirrored)
float measToMps(int16_t meas, bool invert) {
  float rpm = (float)meas;
  if (invert) rpm = -rpm;
  rpm *= SPEED_MEAS_TO_RPM;
  return (rpm / 60.0f) * WHEEL_CIRCUM_M;
}

float measToRpm(int16_t meas, bool invert) {
  float rpm = (float)meas;
  if (invert) rpm = -rpm;
  return rpm * SPEED_MEAS_TO_RPM;
}

// ########################## PRINT ##########################
void printSpeeds(unsigned long timeNow) {
  if (timeNow - iTimePrint < PRINT_MS) return;
  iTimePrint = timeNow;

  float rpmL = measToRpm(Feedback.speedL_meas, true);
  float rpmR = measToRpm(Feedback.speedR_meas, false);
  float vL   = measToMps(Feedback.speedL_meas, true);
  float vR   = measToMps(Feedback.speedR_meas, false);

  Serial.print(F("raw L=")); Serial.print(Feedback.speedL_meas);
  Serial.print(F(" R="));    Serial.print(Feedback.speedR_meas);
  Serial.print(F("  |  rpm L=")); Serial.print(rpmL, 1);
  Serial.print(F(" R="));    Serial.print(rpmR, 1);
  Serial.print(F("  |  m/s L=")); Serial.print(vL, 3);
  Serial.print(F(" R="));    Serial.print(vR, 3);
  Serial.print(F("  |  bat=")); Serial.print(Feedback.batVoltage);
  Serial.print(F("  temp="));   Serial.println(Feedback.boardTemp);
}

// ########################## LOOP ##########################
void loop() {
  unsigned long now = millis();

  // Send forward command
  if (iTimeSend <= now) {
    iTimeSend = now + TIME_SEND;
    Send(0, -SPEED);     // minus = hardware direction fix
  }

  // Receive feedback
  Receive();

  // Print converted speeds
  printSpeeds(now);

  // Heartbeat
  digitalWrite(LED_BUILTIN, (now % 2000) < 1000);
}