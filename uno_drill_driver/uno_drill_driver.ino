// ======================// ======================// ======================
// Drill Controller by Zeke
// ======================// ======================// ======================

#include "DrillDriver.h"

DrillDriver TheDriver(/*direction*/ 10, /*speed*/ 9);

void setup() {
  // For reasons beyond me, it is important that this get called in setup.
  // Attempting to attach the servos in the global constructor causes weird
  // behavior.
  TheDriver.init();
  Serial.begin(9600);
}

void loop() {
  if (Serial.available())
  {
    char input = Serial.read();

    switch(input)
    {
      case 'L' || 'l':
        TheDriver.stopDrill();
        delay(100);
        TheDriver.turnLeft();
        break;
      case 'R' || 'r':
        TheDriver.stopDrill();
        delay(100);
        TheDriver.turnRight();
        break;
      case 'D':
        TheDriver.driveDrill();
        delay(200);
        TheDriver.stopDrill();
        break;
      case 'd':
        TheDriver.driveDrill();
        delay(100);
        TheDriver.stopDrill();
        break;
      case 's':
        TheDriver.stopDrill();
        break;
    }
  }
}
