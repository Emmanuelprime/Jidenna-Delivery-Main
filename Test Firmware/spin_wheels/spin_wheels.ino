// *******************************************************************
//  Arduino Nano — Hoverboard Individual Wheel Test
//  Based on EmanuelFeru/hoverboard-firmware-hack-FOC example
//  Direction fix: both motors inverted (as observed on hardware)
// *******************************************************************

// ########################## DEFINES ##########################
#define HOVER_SERIAL_BAUD   115200
#define SERIAL_BAUD         115200
#define START_FRAME         0xABCD
#define TIME_SEND           100       // [ms] command send interval

#include <SoftwareSerial.h>
SoftwareSerial HoverSerial(2, 3);     // RX, TX

// ########################## STATE ##########################
uint8_t  idx = 0;
uint16_t bufStartFrame;
byte    *p;
byte     incomingByte;
byte     incomingBytePrev;

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

// ########################## TEST STATE ##########################
int  testIndex = 0;
int  testMag   = 150;         // magnitude for each test

// Desired wheel commands (positive = FORWARD on the robot)
int16_t cmdLeft  = 0;
int16_t cmdRight = 0;

// ########################## SETUP ##########################
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println(F("Hoverboard Individual Wheel Test v1.1"));
  Serial.println(F("Commands:"));
  Serial.println(F("  0 = stop both"));
  Serial.println(F("  1 = left forward"));
  Serial.println(F("  2 = right forward"));
  Serial.println(F("  3 = left reverse"));
  Serial.println(F("  4 = right reverse"));
  Serial.println(F("  5 = both forward"));
  Serial.println(F("  6 = both reverse"));
  Serial.println(F("  7 = spin in place (L fwd, R rev)"));
  Serial.println(F("  8 = spin in place (L rev, R fwd)"));
  Serial.println(F("  9 = ramp both forward (auto)"));
  Serial.println(F("Type a number and press Enter."));

  HoverSerial.begin(HOVER_SERIAL_BAUD);
  pinMode(LED_BUILTIN, OUTPUT);
}

// ########################## SEND ##########################
void Send(int16_t uSteer, int16_t uSpeed) {
  Command.start    = (uint16_t)START_FRAME;
  Command.steer    = (int16_t)uSteer;
  Command.speed    = (int16_t)uSpeed;
  Command.checksum = (uint16_t)(Command.start ^ Command.steer ^ Command.speed);
  HoverSerial.write((uint8_t *)&Command, sizeof(Command));
}

// ########################## WHEEL MIXING ##########################
// Physical direction fix:
//   On this hardware, firmware "positive" spins the wheel BACKWARD,
//   so we negate left & right before mixing.
//
// Firmware mixing assumed:  cmdL = speed - steer,  cmdR = speed + steer
// So we solve:              speed = (L + R) / 2,  steer = (R - L) / 2
//
// After the negation, +left means "left wheel forward on the robot".
void SendWheels(int16_t left, int16_t right) {
  // Direction fix — flip both wheels
  left  = -left;
  right = -right;

  int16_t speed = (left + right) / 2;
  int16_t steer = (right - left) / 2;

  Serial.print(F("TX  L=")); Serial.print(left);
  Serial.print(F("  R="));   Serial.print(right);
  Serial.print(F("  -> steer=")); Serial.print(steer);
  Serial.print(F("  speed="));   Serial.println(speed);

  Send(steer, speed);
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
      Serial.print(F("RX  1:")); Serial.print(Feedback.cmd1);
      Serial.print(F("  2:"));   Serial.print(Feedback.cmd2);
      Serial.print(F("  3:R=")); Serial.print(Feedback.speedR_meas);
      Serial.print(F("  4:L=")); Serial.print(Feedback.speedL_meas);
      Serial.print(F("  5:"));   Serial.print(Feedback.batVoltage);
      Serial.print(F("  6:"));   Serial.print(Feedback.boardTemp);
      Serial.print(F("  7:"));   Serial.println(Feedback.cmdLed);
    }
    idx = 0;
  }

  incomingBytePrev = incomingByte;
}

// ########################## TEST SELECTION ##########################
void applyTest(int t) {
  cmdLeft  = 0;
  cmdRight = 0;
  switch (t) {
    case 0: cmdLeft = 0;         cmdRight = 0;         break;
    case 1: cmdLeft = testMag;   cmdRight = 0;         break;
    case 2: cmdLeft = 0;         cmdRight = testMag;   break;
    case 3: cmdLeft = -testMag;  cmdRight = 0;         break;
    case 4: cmdLeft = 0;         cmdRight = -testMag;  break;
    case 5: cmdLeft = testMag;   cmdRight = testMag;   break;
    case 6: cmdLeft = -testMag;  cmdRight = -testMag;  break;
    case 7: cmdLeft = testMag;   cmdRight = -testMag;  break;
    case 8: cmdLeft = -testMag;  cmdRight = testMag;   break;
    case 9: /* handled in loop() as ramp */            break;
    default: break;
  }
}

// ########################## LOOP ##########################
unsigned long iTimeSend = 0;
unsigned long iTimeRamp = 0;
int  rampValue = 0;
int  rampStep  = 20;

void loop(void) {
  unsigned long timeNow = millis();

  // Handle Serial Monitor input
  if (Serial.available()) {
    int c = Serial.read();
    if (c >= '0' && c <= '9') {
      testIndex = c - '0';
      applyTest(testIndex);
      if (testIndex == 9) {
        rampValue = 0;
        rampStep  = 20;
      }
      Serial.print(F(">>> Test ")); Serial.println(testIndex);
    }
  }

  // Update ramp test
  if (testIndex == 9 && (timeNow - iTimeRamp) >= TIME_SEND) {
    iTimeRamp = timeNow;
    rampValue += rampStep;
    if (rampValue >= 300 || rampValue <= -300) rampStep = -rampStep;
    cmdLeft  = rampValue;
    cmdRight = rampValue;
  }

  // Receive from hoverboard
  Receive();

  // Send commands
  if (iTimeSend > timeNow) return;
  iTimeSend = timeNow + TIME_SEND;

  SendWheels(cmdLeft, cmdRight);

  digitalWrite(LED_BUILTIN, (timeNow % 2000) < 1000);
}