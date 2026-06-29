// ============================================================
//  relay_controller.h — Relay Zone State Management
//  Digital Migraine Therapy Helmet
// ============================================================

#ifndef RELAY_CONTROLLER_H
#define RELAY_CONTROLLER_H

#include "config.h"

// ------------------------------------------------------------
// Zone Index Enum
// ------------------------------------------------------------
enum ZoneID {
  ZONE_FRONTAL      = 0,
  ZONE_PARIETAL     = 1,
  ZONE_OCCIPITAL    = 2,
  ZONE_RIGHT_TEMP   = 3,
  ZONE_LEFT_TEMP    = 4,
  ZONE_FRONTAL_PAD  = 5,
  ZONE_OCCIPITAL_PAD= 6,
  ZONE_COUNT        = 7
};

// ------------------------------------------------------------
// Actuator Zone Struct
// ------------------------------------------------------------
struct ActuatorZone {
  uint8_t  relayPin;
  bool     enabled;
  uint8_t  intensityLevel;      // 0–10
  unsigned long onDurationMs;   // how long relay stays ON per cycle
  unsigned long offDurationMs;  // how long relay stays OFF per cycle
  unsigned long totalDurationMs;// total active time for this zone
  unsigned long startedAt;      // millis() when zone was activated
  unsigned long lastToggleAt;   // millis() of last relay toggle
  bool     relayState;          // true = ON
  bool     finished;            // true when totalDurationMs elapsed
  char     technique[24];       // e.g. "SUSTAINED_PRESSURE"
};

// ------------------------------------------------------------
// Temperature Pad Struct
// ------------------------------------------------------------
struct TempPad {
  uint8_t  heatPin;
  uint8_t  coldPin;
  uint8_t  masterPin;
  bool     enabled;
  uint8_t  mode;                // PAD_OFF / PAD_HEAT / PAD_COLD / PAD_ALTERNATING
  unsigned long totalDurationMs;
  unsigned long startedAt;
  unsigned long lastToggleAt;
  bool     altPhaseHeat;        // true = currently in heat phase of alternating
  unsigned long altHeatMs;      // heat phase duration ms
  unsigned long altColdMs;      // cold phase duration ms
  int      altRepeatCount;      // remaining repeats (-1 = infinite)
  bool     finished;
};

// ------------------------------------------------------------
// Global Zone Arrays
// ------------------------------------------------------------
extern ActuatorZone actuators[5];
extern TempPad      tempPads[2];

// ------------------------------------------------------------
// Function Declarations
// ------------------------------------------------------------
void initRelays();
void resetAllZones();
void allRelaysOff();

void setActuator(ZoneID id, bool enabled, uint8_t intensity,
                 unsigned long onMs, unsigned long offMs,
                 unsigned long totalMs, const char* technique);

void setTempPad(ZoneID id, bool enabled, uint8_t mode,
                unsigned long totalMs,
                unsigned long altHeatMs, unsigned long altColdMs,
                int altRepeats);

void tickActuators();
void tickTempPads();

#endif // RELAY_CONTROLLER_H
