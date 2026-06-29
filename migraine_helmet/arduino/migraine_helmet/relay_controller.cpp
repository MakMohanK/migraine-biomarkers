// ============================================================
//  relay_controller.cpp — Relay Zone Implementation
//  Digital Migraine Therapy Helmet
// ============================================================

#include "relay_controller.h"
#include "config.h"

// ------------------------------------------------------------
// Global Zone Instances
// ------------------------------------------------------------
ActuatorZone actuators[5];
TempPad      tempPads[2];

// ------------------------------------------------------------
//  initRelays()
//  Set all relay pins as OUTPUT and turn them all OFF
// ------------------------------------------------------------
void initRelays() {
  // Actuator relay pins
  uint8_t actPins[5] = {
    RELAY_FRONTAL, RELAY_PARIETAL, RELAY_OCCIPITAL,
    RELAY_RIGHT_TEMP, RELAY_LEFT_TEMP
  };
  for (int i = 0; i < 5; i++) {
    pinMode(actPins[i], OUTPUT);
    digitalWrite(actPins[i], RELAY_OFF);
  }

  // Temperature pad relay pins
  uint8_t padPins[6] = {
    RELAY_FRONTAL_PAD_HEAT,   RELAY_FRONTAL_PAD_COLD,   RELAY_FRONTAL_PAD_MASTER,
    RELAY_OCCIPITAL_PAD_HEAT, RELAY_OCCIPITAL_PAD_COLD, RELAY_OCCIPITAL_PAD_MASTER
  };
  for (int i = 0; i < 6; i++) {
    pinMode(padPins[i], OUTPUT);
    digitalWrite(padPins[i], RELAY_OFF);
  }

  // Status LED
  pinMode(LED_STATUS, OUTPUT);
  digitalWrite(LED_STATUS, LOW);

  // Init actuator structs
  uint8_t aRelayMap[5] = {
    RELAY_FRONTAL, RELAY_PARIETAL, RELAY_OCCIPITAL,
    RELAY_RIGHT_TEMP, RELAY_LEFT_TEMP
  };
  for (int i = 0; i < 5; i++) {
    actuators[i].relayPin       = aRelayMap[i];
    actuators[i].enabled        = false;
    actuators[i].intensityLevel = 0;
    actuators[i].onDurationMs   = 0;
    actuators[i].offDurationMs  = 500;
    actuators[i].totalDurationMs= 0;
    actuators[i].startedAt      = 0;
    actuators[i].lastToggleAt   = 0;
    actuators[i].relayState     = false;
    actuators[i].finished       = true;
    strncpy(actuators[i].technique, "NONE", 24);
  }

  // Init temp pad structs
  tempPads[0].heatPin    = RELAY_FRONTAL_PAD_HEAT;
  tempPads[0].coldPin    = RELAY_FRONTAL_PAD_COLD;
  tempPads[0].masterPin  = RELAY_FRONTAL_PAD_MASTER;

  tempPads[1].heatPin    = RELAY_OCCIPITAL_PAD_HEAT;
  tempPads[1].coldPin    = RELAY_OCCIPITAL_PAD_COLD;
  tempPads[1].masterPin  = RELAY_OCCIPITAL_PAD_MASTER;

  for (int i = 0; i < 2; i++) {
    tempPads[i].enabled        = false;
    tempPads[i].mode           = PAD_OFF;
    tempPads[i].totalDurationMs= 0;
    tempPads[i].startedAt      = 0;
    tempPads[i].lastToggleAt   = 0;
    tempPads[i].altPhaseHeat   = true;
    tempPads[i].altHeatMs      = 0;
    tempPads[i].altColdMs      = 0;
    tempPads[i].altRepeatCount = 0;
    tempPads[i].finished       = true;
  }
}

// ------------------------------------------------------------
//  allRelaysOff()
//  Emergency / end-of-session — force everything OFF
// ------------------------------------------------------------
void allRelaysOff() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(actuators[i].relayPin, RELAY_OFF);
    actuators[i].relayState = false;
    actuators[i].finished   = true;
  }
  for (int i = 0; i < 2; i++) {
    digitalWrite(tempPads[i].heatPin,   RELAY_OFF);
    digitalWrite(tempPads[i].coldPin,   RELAY_OFF);
    digitalWrite(tempPads[i].masterPin, RELAY_OFF);
    tempPads[i].finished = true;
    tempPads[i].enabled  = false;
  }
  digitalWrite(LED_STATUS, LOW);
}

// ------------------------------------------------------------
//  resetAllZones()
//  Full reset — same as allRelaysOff but also clears structs
// ------------------------------------------------------------
void resetAllZones() {
  allRelaysOff();
  for (int i = 0; i < 5; i++) {
    actuators[i].enabled        = false;
    actuators[i].intensityLevel = 0;
    actuators[i].startedAt      = 0;
    actuators[i].lastToggleAt   = 0;
    strncpy(actuators[i].technique, "NONE", 24);
  }
  for (int i = 0; i < 2; i++) {
    tempPads[i].enabled        = false;
    tempPads[i].mode           = PAD_OFF;
    tempPads[i].altRepeatCount = 0;
  }
}

// ------------------------------------------------------------
//  setActuator()
//  Configure one actuator zone before starting therapy
// ------------------------------------------------------------
void setActuator(ZoneID id, bool enabled, uint8_t intensity,
                 unsigned long onMs, unsigned long offMs,
                 unsigned long totalMs, const char* technique) {
  int idx = (int)id;
  if (idx < 0 || idx > 4) return;

  actuators[idx].enabled         = enabled;
  actuators[idx].intensityLevel  = intensity;
  actuators[idx].onDurationMs    = (intensity == 0) ? 0 : INTENSITY_ON_MS[intensity];
  actuators[idx].offDurationMs   = offMs;
  actuators[idx].totalDurationMs = totalMs;
  actuators[idx].startedAt       = 0;   // set when therapy starts
  actuators[idx].lastToggleAt    = 0;
  actuators[idx].relayState      = false;
  actuators[idx].finished        = !enabled;
  strncpy(actuators[idx].technique, technique, 23);
  actuators[idx].technique[23]   = '\0';

  // If disabled, ensure relay is off
  if (!enabled) {
    digitalWrite(actuators[idx].relayPin, RELAY_OFF);
  }
}

// ------------------------------------------------------------
//  setTempPad()
//  Configure one temperature pad before starting therapy
// ------------------------------------------------------------
void setTempPad(ZoneID id, bool enabled, uint8_t mode,
                unsigned long totalMs,
                unsigned long altHeatMs, unsigned long altColdMs,
                int altRepeats) {
  int idx = (id == ZONE_FRONTAL_PAD) ? 0 : 1;

  tempPads[idx].enabled         = enabled;
  tempPads[idx].mode            = mode;
  tempPads[idx].totalDurationMs = totalMs;
  tempPads[idx].startedAt       = 0;
  tempPads[idx].lastToggleAt    = 0;
  tempPads[idx].altPhaseHeat    = true;
  tempPads[idx].altHeatMs       = altHeatMs;
  tempPads[idx].altColdMs       = altColdMs;
  tempPads[idx].altRepeatCount  = altRepeats;
  tempPads[idx].finished        = !enabled;

  if (!enabled) {
    digitalWrite(tempPads[idx].heatPin,   RELAY_OFF);
    digitalWrite(tempPads[idx].coldPin,   RELAY_OFF);
    digitalWrite(tempPads[idx].masterPin, RELAY_OFF);
  }
}

// ============================================================
//  tickActuators()
//  Called every loop iteration — manages ON/OFF duty cycling
//  for all 5 actuator zones based on their technique & intensity
// ============================================================
void tickActuators() {
  unsigned long now = millis();

  for (int i = 0; i < 5; i++) {
    ActuatorZone& z = actuators[i];

    if (!z.enabled || z.finished) continue;

    // Check total duration expiry
    if (z.totalDurationMs > 0 && (now - z.startedAt) >= z.totalDurationMs) {
      digitalWrite(z.relayPin, RELAY_OFF);
      z.relayState = false;
      z.finished   = true;
      continue;
    }

    // --- Technique-specific relay toggling ---

    // SUSTAINED_PRESSURE: just keep ON for onDuration, OFF for offDuration
    // CIRCULAR / PETRISSAGE / EFFLEURAGE: same duty cycle, different semantics handled by hardware
    // RHYTHMIC_TAPPING / VIBRATION: faster on/off using frequency_hz mapped to onDurationMs
    // TRIGGER_POINT: long ON, short OFF

    unsigned long elapsed = now - z.lastToggleAt;

    if (z.relayState) {
      // Currently ON — check if should turn OFF
      if (elapsed >= z.onDurationMs) {
        digitalWrite(z.relayPin, RELAY_OFF);
        z.relayState    = false;
        z.lastToggleAt  = now;
      }
    } else {
      // Currently OFF — check if should turn ON
      if (elapsed >= z.offDurationMs) {
        if (z.intensityLevel > 0) {
          digitalWrite(z.relayPin, RELAY_ON);
          z.relayState   = true;
          z.lastToggleAt = now;
        }
      }
    }
  }
}

// ============================================================
//  tickTempPads()
//  Called every loop iteration — manages HEAT / COLD / ALTERNATING
//  relay switching for both temperature pads
// ============================================================
void tickTempPads() {
  unsigned long now = millis();

  for (int i = 0; i < 2; i++) {
    TempPad& p = tempPads[i];

    if (!p.enabled || p.finished) continue;

    // Check total duration expiry
    if (p.totalDurationMs > 0 && (now - p.startedAt) >= p.totalDurationMs) {
      digitalWrite(p.heatPin,   RELAY_OFF);
      digitalWrite(p.coldPin,   RELAY_OFF);
      digitalWrite(p.masterPin, RELAY_OFF);
      p.finished = true;
      continue;
    }

    unsigned long elapsed = now - p.lastToggleAt;

    switch (p.mode) {

      // ---- HEAT: master ON + heat relay ON, cold OFF ----
      case PAD_HEAT:
        digitalWrite(p.masterPin, RELAY_ON);
        digitalWrite(p.heatPin,   RELAY_ON);
        digitalWrite(p.coldPin,   RELAY_OFF);
        break;

      // ---- COLD: master ON + cold relay ON, heat OFF ----
      case PAD_COLD:
        digitalWrite(p.masterPin, RELAY_ON);
        digitalWrite(p.coldPin,   RELAY_ON);
        digitalWrite(p.heatPin,   RELAY_OFF);
        break;

      // ---- ALTERNATING: switch between heat and cold phases ----
      case PAD_ALTERNATING:
        if (p.altPhaseHeat) {
          // In heat phase
          digitalWrite(p.masterPin, RELAY_ON);
          digitalWrite(p.heatPin,   RELAY_ON);
          digitalWrite(p.coldPin,   RELAY_OFF);
          if (elapsed >= p.altHeatMs) {
            // Switch to cold phase
            p.altPhaseHeat = false;
            p.lastToggleAt = now;
            if (p.altRepeatCount > 0) p.altRepeatCount--;
          }
        } else {
          // In cold phase
          digitalWrite(p.masterPin, RELAY_ON);
          digitalWrite(p.coldPin,   RELAY_ON);
          digitalWrite(p.heatPin,   RELAY_OFF);
          if (elapsed >= p.altColdMs) {
            if (p.altRepeatCount == 0) {
              // Done all repeats
              p.finished = true;
              digitalWrite(p.heatPin,   RELAY_OFF);
              digitalWrite(p.coldPin,   RELAY_OFF);
              digitalWrite(p.masterPin, RELAY_OFF);
            } else {
              // Switch back to heat phase
              p.altPhaseHeat = true;
              p.lastToggleAt = now;
            }
          }
        }
        break;

      // ---- OFF ----
      case PAD_OFF:
      default:
        digitalWrite(p.heatPin,   RELAY_OFF);
        digitalWrite(p.coldPin,   RELAY_OFF);
        digitalWrite(p.masterPin, RELAY_OFF);
        break;
    }
  }
}
