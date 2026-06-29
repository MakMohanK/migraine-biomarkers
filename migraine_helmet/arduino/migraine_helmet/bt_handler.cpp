// ============================================================
//  bt_handler.cpp — Bluetooth Command Handler Implementation
//  Digital Migraine Therapy Helmet
// ============================================================

#include "bt_handler.h"
#include "relay_controller.h"
#include "config.h"

#if USE_MEGA
  #define BT_SERIAL Serial1
#else
  #include <SoftwareSerial.h>
  SoftwareSerial btSerial(10, 11); // RX, TX
  #define BT_SERIAL btSerial
#endif

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
SessionInfo  currentSession;
SessionState sessionState = STATE_IDLE;
unsigned long lastBtRxAt  = 0;

static String btBuffer = "";

// ------------------------------------------------------------
//  btInit()
// ------------------------------------------------------------
void btInit() {
  BT_SERIAL.begin(BT_BAUD);
  memset(&currentSession, 0, sizeof(currentSession));
  sessionState = STATE_IDLE;
  lastBtRxAt   = millis();
}

// ------------------------------------------------------------
//  btSend()  — send a raw string over Bluetooth
// ------------------------------------------------------------
void btSend(const char* msg) {
  BT_SERIAL.print(msg);
  BT_SERIAL.print("\r\n");
}

// ------------------------------------------------------------
//  btSendAck()  — send a structured ACK/NACK JSON string
// ------------------------------------------------------------
void btSendAck(const char* cmd, const char* status, const char* msg) {
  String ack = "{\"ack\":\"";
  ack += cmd;
  ack += "\",\"status\":\"";
  ack += status;
  ack += "\",\"msg\":\"";
  ack += msg;
  ack += "\"}";
  btSend(ack.c_str());
}

// ------------------------------------------------------------
//  btTick()  — read incoming bytes, assemble lines, dispatch
// ------------------------------------------------------------
void btTick() {
  while (BT_SERIAL.available()) {
    char c = BT_SERIAL.read();
    lastBtRxAt = millis();

    if (c == '\n' || c == '\r') {
      btBuffer.trim();
      if (btBuffer.length() > 0) {
        processCommand(btBuffer);
        btBuffer = "";
      }
    } else {
      btBuffer += c;
      // Guard against buffer overflow
      if (btBuffer.length() > 480) btBuffer = "";
    }
  }

  // BT signal loss watchdog
  if (sessionState == STATE_RUNNING) {
    if ((millis() - lastBtRxAt) > BT_TIMEOUT_MS) {
      sessionState = STATE_PAUSED;
      allRelaysOff();
      btSendAck("WATCHDOG", "PAUSED", "BT signal lost - session paused");
    }
  }
}

// ------------------------------------------------------------
//  processCommand()  — route to correct handler
// ------------------------------------------------------------
void processCommand(const String& raw) {
  String cmd = extractField(raw, "cmd");

  if      (cmd == "CMD_HANDSHAKE")      handleHandshake(raw);
  else if (cmd == "CMD_LOAD_SESSION")   handleLoadSession(raw);
  else if (cmd == "CMD_ACTUATOR_SET")   handleActuatorSet(raw);
  else if (cmd == "CMD_TEMP_PAD_SET")   handleTempPadSet(raw);
  else if (cmd == "CMD_START_THERAPY")  handleStartTherapy(raw);
  else if (cmd == "CMD_PAUSE_THERAPY")  handlePauseTherapy(raw);
  else if (cmd == "CMD_RESUME_THERAPY") handleResumeTherapy(raw);
  else if (cmd == "CMD_STOP_THERAPY")   handleStopTherapy(raw);
  else if (cmd == "CMD_STATUS_REQUEST") handleStatusRequest();
  else if (cmd == "CMD_EMERGENCY_STOP") handleEmergencyStop();
  else {
    btSendAck("UNKNOWN", "NACK", "Unknown command");
  }
}

// ============================================================
//  COMMAND HANDLERS
// ============================================================

// ------------------------------------------------------------
//  CMD_HANDSHAKE
// ------------------------------------------------------------
void handleHandshake(const String& raw) {
  String resp = "{\"status\":\"ACK\",\"device_id\":\"HELMET_001\","
                "\"firmware_version\":\"1.0.0\",\"battery_percent\":100}";
  btSend(resp.c_str());
}

// ------------------------------------------------------------
//  CMD_LOAD_SESSION
// ------------------------------------------------------------
void handleLoadSession(const String& raw) {
  if (sessionState == STATE_RUNNING) {
    btSendAck("CMD_LOAD_SESSION", "NACK", "Session already running");
    return;
  }

  String sid = extractField(raw, "session_id");
  String pid = extractField(raw, "patient_id");
  String prf = extractField(raw, "therapy_profile");
  long   dur = extractLong(raw, "total_duration_min");

  sid.toCharArray(currentSession.sessionId, 32);
  pid.toCharArray(currentSession.patientId, 16);
  prf.toCharArray(currentSession.therapyProfile, 12);
  currentSession.totalDurationMs = (uint32_t)(dur * 60000UL);
  currentSession.totalPausedMs   = 0;
  currentSession.pausedAt        = 0;
  currentSession.state           = STATE_LOADED;

  sessionState = STATE_LOADED;
  resetAllZones();

  btSendAck("CMD_LOAD_SESSION", "ACK", currentSession.sessionId);
}

// ------------------------------------------------------------
//  CMD_ACTUATOR_SET
// ------------------------------------------------------------
void handleActuatorSet(const String& raw) {
  String zoneStr = extractField(raw, "zone");
  bool   enabled = extractBool(raw, "enabled");
  long   intLvl  = extractLong(raw, "intensity_level");
  long   onSec   = extractLong(raw, "on_seconds");
  long   offSec  = extractLong(raw, "off_seconds");
  long   durSec  = extractLong(raw, "duration_seconds");
  String tech    = extractField(raw, "technique");

  unsigned long onMs  = (unsigned long)(onSec  * 1000UL);
  unsigned long offMs = (unsigned long)(offSec  * 1000UL);
  unsigned long durMs = (unsigned long)(durSec  * 1000UL);

  uint8_t intensity = (uint8_t)constrain(intLvl, 0, 10);

  if (zoneStr == "ALL") {
    for (int i = 0; i <= 4; i++) {
      setActuator((ZoneID)i, enabled, intensity, onMs, offMs, durMs, tech.c_str());
    }
    btSendAck("CMD_ACTUATOR_SET", "ACK", "ALL zones configured");
  } else {
    ZoneID zid = parseZoneID(zoneStr);
    if (zid >= ZONE_COUNT) {
      btSendAck("CMD_ACTUATOR_SET", "NACK", "Unknown zone");
      return;
    }
    setActuator(zid, enabled, intensity, onMs, offMs, durMs, tech.c_str());
    btSendAck("CMD_ACTUATOR_SET", "ACK", zoneStr.c_str());
  }
}

// ------------------------------------------------------------
//  CMD_TEMP_PAD_SET
// ------------------------------------------------------------
void handleTempPadSet(const String& raw) {
  String  padStr   = extractField(raw, "pad");
  bool    enabled  = extractBool(raw,  "enabled");
  String  modeStr  = extractField(raw, "mode");
  long    durSec   = extractLong(raw,  "duration_seconds");
  long    heatSec  = extractLong(raw,  "heat_duration_seconds");
  long    coldSec  = extractLong(raw,  "cold_duration_seconds");
  long    repeats  = extractLong(raw,  "repeat_count");

  uint8_t       mode  = parsePadMode(modeStr);
  unsigned long durMs  = (unsigned long)(durSec  * 1000UL);
  unsigned long heatMs = (unsigned long)(heatSec * 1000UL);
  unsigned long coldMs = (unsigned long)(coldSec * 1000UL);

  ZoneID padID = (padStr == "frontal_pad") ? ZONE_FRONTAL_PAD : ZONE_OCCIPITAL_PAD;

  setTempPad(padID, enabled, mode, durMs, heatMs, coldMs, (int)repeats);
  btSendAck("CMD_TEMP_PAD_SET", "ACK", padStr.c_str());
}

// ------------------------------------------------------------
//  CMD_START_THERAPY
// ------------------------------------------------------------
void handleStartTherapy(const String& raw) {
  String sid = extractField(raw, "session_id");

  if (sessionState != STATE_LOADED) {
    btSendAck("CMD_START_THERAPY", "NACK", "Load session first");
    return;
  }

  unsigned long now = millis();
  currentSession.startedAt = now;

  // Arm all enabled actuators
  for (int i = 0; i < 5; i++) {
    if (actuators[i].enabled) {
      actuators[i].startedAt    = now;
      actuators[i].lastToggleAt = now;
      actuators[i].relayState   = false;
      actuators[i].finished     = false;
    }
  }

  // Arm all enabled temp pads
  for (int i = 0; i < 2; i++) {
    if (tempPads[i].enabled) {
      tempPads[i].startedAt    = now;
      tempPads[i].lastToggleAt = now;
      tempPads[i].finished     = false;
    }
  }

  sessionState = STATE_RUNNING;
  digitalWrite(LED_STATUS, HIGH);

  btSendAck("CMD_START_THERAPY", "ACK", "Therapy started");
}

// ------------------------------------------------------------
//  CMD_PAUSE_THERAPY
// ------------------------------------------------------------
void handlePauseTherapy(const String& raw) {
  if (sessionState != STATE_RUNNING) {
    btSendAck("CMD_PAUSE_THERAPY", "NACK", "Not running");
    return;
  }

  allRelaysOff();
  currentSession.pausedAt = millis();
  sessionState = STATE_PAUSED;
  digitalWrite(LED_STATUS, LOW);

  unsigned long elapsed = (millis() - currentSession.startedAt - currentSession.totalPausedMs) / 1000UL;
  String msg = "Elapsed:" + String(elapsed) + "s";
  btSendAck("CMD_PAUSE_THERAPY", "ACK", msg.c_str());
}

// ------------------------------------------------------------
//  CMD_RESUME_THERAPY
// ------------------------------------------------------------
void handleResumeTherapy(const String& raw) {
  if (sessionState != STATE_PAUSED) {
    btSendAck("CMD_RESUME_THERAPY", "NACK", "Not paused");
    return;
  }

  unsigned long pauseDuration = millis() - currentSession.pausedAt;
  currentSession.totalPausedMs += pauseDuration;

  // Shift all zone start times by pause duration so they resume correctly
  for (int i = 0; i < 5; i++) {
    if (!actuators[i].finished) {
      actuators[i].startedAt    += pauseDuration;
      actuators[i].lastToggleAt  = millis();
    }
  }
  for (int i = 0; i < 2; i++) {
    if (!tempPads[i].finished) {
      tempPads[i].startedAt    += pauseDuration;
      tempPads[i].lastToggleAt  = millis();
    }
  }

  sessionState = STATE_RUNNING;
  digitalWrite(LED_STATUS, HIGH);

  unsigned long remaining = 0;
  if (currentSession.totalDurationMs > 0) {
    unsigned long active = millis() - currentSession.startedAt - currentSession.totalPausedMs;
    if (active < currentSession.totalDurationMs)
      remaining = (currentSession.totalDurationMs - active) / 1000UL;
  }

  String msg = "Remaining:" + String(remaining) + "s";
  btSendAck("CMD_RESUME_THERAPY", "ACK", msg.c_str());
}

// ------------------------------------------------------------
//  CMD_STOP_THERAPY
// ------------------------------------------------------------
void handleStopTherapy(const String& raw) {
  allRelaysOff();

  unsigned long totalActive = 0;
  if (currentSession.startedAt > 0) {
    totalActive = (millis() - currentSession.startedAt - currentSession.totalPausedMs) / 1000UL;
  }

  sessionState = STATE_FINISHED;
  memset(&currentSession, 0, sizeof(currentSession));
  sessionState = STATE_IDLE;
  digitalWrite(LED_STATUS, LOW);

  String msg = "TotalActive:" + String(totalActive) + "s";
  btSendAck("CMD_STOP_THERAPY", "ACK", msg.c_str());
}

// ------------------------------------------------------------
//  CMD_STATUS_REQUEST
// ------------------------------------------------------------
void handleStatusRequest() {
  String stateStr;
  switch (sessionState) {
    case STATE_IDLE:     stateStr = "IDLE";     break;
    case STATE_LOADED:   stateStr = "LOADED";   break;
    case STATE_RUNNING:  stateStr = "RUNNING";  break;
    case STATE_PAUSED:   stateStr = "PAUSED";   break;
    case STATE_FINISHED: stateStr = "FINISHED"; break;
    default:             stateStr = "ERROR";    break;
  }

  unsigned long elapsed   = 0;
  unsigned long remaining = 0;
  if (currentSession.startedAt > 0) {
    elapsed = (millis() - currentSession.startedAt - currentSession.totalPausedMs) / 1000UL;
    if (currentSession.totalDurationMs > 0 && (elapsed * 1000UL) < currentSession.totalDurationMs)
      remaining = (currentSession.totalDurationMs / 1000UL) - elapsed;
  }

  // Build actuator status
  const char* zoneName[5] = {"frontal_lobe","parietal_lobe","occipital_lobe","right_temporal","left_temporal"};
  String actStr = "";
  for (int i = 0; i < 5; i++) {
    actStr += "\"";
    actStr += zoneName[i];
    actStr += "\":{\"active\":";
    actStr += actuators[i].relayState ? "true" : "false";
    actStr += ",\"intensity\":";
    actStr += actuators[i].intensityLevel;
    actStr += ",\"technique\":\"";
    actStr += actuators[i].technique;
    actStr += "\"}";
    if (i < 4) actStr += ",";
  }

  // Build pad status
  const char* padModeStr[4] = {"OFF","HEAT","COLD","ALTERNATING"};
  const char* padName[2]    = {"frontal_pad","occipital_pad"};
  String padStr = "";
  for (int i = 0; i < 2; i++) {
    padStr += "\"";
    padStr += padName[i];
    padStr += "\":{\"active\":";
    padStr += tempPads[i].enabled ? "true" : "false";
    padStr += ",\"mode\":\"";
    padStr += padModeStr[tempPads[i].mode];
    padStr += "\"}";
    if (i < 1) padStr += ",";
  }

  String resp = "{\"status\":\"" + stateStr + "\","
                "\"session_id\":\"" + String(currentSession.sessionId) + "\","
                "\"elapsed_s\":" + String(elapsed) + ","
                "\"remaining_s\":" + String(remaining) + ","
                "\"actuators\":{" + actStr + "},"
                "\"temp_pads\":{" + padStr + "}}";

  btSend(resp.c_str());
}

// ------------------------------------------------------------
//  CMD_EMERGENCY_STOP
// ------------------------------------------------------------
void handleEmergencyStop() {
  allRelaysOff();
  sessionState = STATE_IDLE;
  memset(&currentSession, 0, sizeof(currentSession));
  digitalWrite(LED_STATUS, LOW);
  btSendAck("CMD_EMERGENCY_STOP", "ACK", "ALL_ZONES_OFF");
}

// ============================================================
//  HELPER PARSERS
// ============================================================

// Simple key extract from flat JSON string: {"key":"value",...}
String extractField(const String& json, const String& key) {
  String search = "\"" + key + "\"";
  int idx = json.indexOf(search);
  if (idx < 0) return "";

  int colon = json.indexOf(':', idx + search.length());
  if (colon < 0) return "";

  int start = colon + 1;
  while (start < (int)json.length() && json[start] == ' ') start++;

  if (json[start] == '"') {
    int end = json.indexOf('"', start + 1);
    if (end < 0) return "";
    return json.substring(start + 1, end);
  } else {
    // numeric / bool / null
    int end = start;
    while (end < (int)json.length() &&
           json[end] != ',' && json[end] != '}' && json[end] != ']') end++;
    return json.substring(start, end);
  }
}

long extractLong(const String& json, const String& key) {
  String val = extractField(json, key);
  if (val.length() == 0 || val == "null") return 0;
  return val.toInt();
}

bool extractBool(const String& json, const String& key) {
  String val = extractField(json, key);
  val.toLowerCase();
  return (val == "true" || val == "1");
}

ZoneID parseZoneID(const String& z) {
  if (z == "frontal_lobe")    return ZONE_FRONTAL;
  if (z == "parietal_lobe")   return ZONE_PARIETAL;
  if (z == "occipital_lobe")  return ZONE_OCCIPITAL;
  if (z == "right_temporal")  return ZONE_RIGHT_TEMP;
  if (z == "left_temporal")   return ZONE_LEFT_TEMP;
  if (z == "frontal_pad")     return ZONE_FRONTAL_PAD;
  if (z == "occipital_pad")   return ZONE_OCCIPITAL_PAD;
  return ZONE_COUNT; // invalid
}

uint8_t parsePadMode(const String& m) {
  if (m == "HEAT")        return PAD_HEAT;
  if (m == "COLD")        return PAD_COLD;
  if (m == "ALTERNATING") return PAD_ALTERNATING;
  return PAD_OFF;
}
