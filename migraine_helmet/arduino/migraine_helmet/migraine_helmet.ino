// ============================================================
//  migraine_helmet.ino — Main Entry Point
//  Digital Migraine Therapy Helmet
//  Board : Arduino Mega 2560 (recommended) / Uno
//  HC-05 : Connected to Serial1 (Mega) or SoftwareSerial (Uno)
//
//  RELAY WIRING SUMMARY
//  ---------------------------------------------------------------
//  Pin 2  → Relay 1  → Frontal Lobe Actuator
//  Pin 3  → Relay 2  → Parietal Lobe Actuator
//  Pin 4  → Relay 3  → Occipital Lobe Actuator
//  Pin 5  → Relay 4  → Right Temporal Actuator (near right ear)
//  Pin 6  → Relay 5  → Left Temporal Actuator  (near left ear)
//  Pin 7  → Relay 6  → Frontal Pad  — HEAT element
//  Pin 8  → Relay 7  → Frontal Pad  — COLD element (Peltier)
//  Pin 9  → Relay 8  → Frontal Pad  — MASTER power
//  Pin 10 → Relay 9  → Occipital Pad — HEAT element
//  Pin 11 → Relay 10 → Occipital Pad — COLD element (Peltier)
//  Pin 12 → Relay 11 → Occipital Pad — MASTER power
//  Pin 13 → Onboard LED (heartbeat / session active indicator)
//
//  HC-05 WIRING (Mega)
//  ---------------------------------------------------------------
//  HC-05 TX → Arduino Mega RX1 (Pin 19)
//  HC-05 RX → Arduino Mega TX1 (Pin 18) via 1kΩ voltage divider
//  HC-05 VCC → 5V
//  HC-05 GND → GND
// ============================================================

#include "config.h"
#include "relay_controller.h"
#include "bt_handler.h"

// ------------------------------------------------------------
//  Heartbeat LED blink (non-blocking)
// ------------------------------------------------------------
static unsigned long lastHeartbeat = 0;
static bool          ledState      = false;

void heartbeatTick() {
  // Fast blink when running, slow blink when idle/paused
  unsigned long interval = (sessionState == STATE_RUNNING) ? 300UL : 1000UL;
  if ((millis() - lastHeartbeat) >= interval) {
    ledState = !ledState;
    digitalWrite(LED_STATUS, ledState ? HIGH : LOW);
    lastHeartbeat = millis();
  }
}

// ------------------------------------------------------------
//  Session completion checker
// ------------------------------------------------------------
void checkSessionCompletion() {
  if (sessionState != STATE_RUNNING) return;
  if (currentSession.totalDurationMs == 0) return;

  unsigned long activeMs = millis()
                         - currentSession.startedAt
                         - currentSession.totalPausedMs;

  if (activeMs >= currentSession.totalDurationMs) {
    // Auto-complete the session
    allRelaysOff();
    sessionState = STATE_FINISHED;
    btSendAck("SESSION", "COMPLETED", currentSession.sessionId);
    memset(&currentSession, 0, sizeof(currentSession));
    sessionState = STATE_IDLE;
  }
}

// ------------------------------------------------------------
//  All-zones-finished checker
//  If every actuator and pad has finished early, end session
// ------------------------------------------------------------
void checkAllZonesFinished() {
  if (sessionState != STATE_RUNNING) return;

  for (int i = 0; i < 5; i++) {
    if (actuators[i].enabled && !actuators[i].finished) return;
  }
  for (int i = 0; i < 2; i++) {
    if (tempPads[i].enabled && !tempPads[i].finished) return;
  }

  // All zones done
  allRelaysOff();
  sessionState = STATE_FINISHED;
  btSendAck("SESSION", "COMPLETED", "All zones finished");
  memset(&currentSession, 0, sizeof(currentSession));
  sessionState = STATE_IDLE;
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  // Debug serial (USB)
  Serial.begin(115200);
  Serial.println(F("=== Migraine Therapy Helmet v1.0 ==="));
  Serial.println(F("Initialising relays..."));

  // Initialise all relay pins and zero out zone structs
  initRelays();

  // Initialise Bluetooth handler
  btInit();

  Serial.println(F("Bluetooth ready. Waiting for connection..."));
  Serial.println(F("Ready."));
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  // 1. Read & dispatch incoming Bluetooth commands
  btTick();

  // 2. If session is running, tick all relay zones
  if (sessionState == STATE_RUNNING) {
    tickActuators();
    tickTempPads();
    checkSessionCompletion();
    checkAllZonesFinished();
  }

  // 3. Heartbeat LED
  heartbeatTick();

  // 4. Small yield — keeps loop responsive without blocking
  delay(10);
}
