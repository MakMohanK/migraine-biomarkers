// ============================================================
//  bt_handler.h — Bluetooth Command Handler (HC-05)
//  Digital Migraine Therapy Helmet
// ============================================================

#ifndef BT_HANDLER_H
#define BT_HANDLER_H

#include "config.h"
#include "relay_controller.h"

// ------------------------------------------------------------
// Session State Machine
// ------------------------------------------------------------
enum SessionState {
  STATE_IDLE      = 0,
  STATE_LOADED    = 1,
  STATE_RUNNING   = 2,
  STATE_PAUSED    = 3,
  STATE_FINISHED  = 4,
  STATE_ERROR     = 5
};

// ------------------------------------------------------------
// Session Info (loaded from CMD_LOAD_SESSION)
// ------------------------------------------------------------
struct SessionInfo {
  char     sessionId[32];
  char     patientId[16];
  char     therapyProfile[12];  // gentle / moderate / intensive / custom
  uint32_t totalDurationMs;
  uint32_t startedAt;
  uint32_t pausedAt;
  uint32_t totalPausedMs;
  SessionState state;
};

// ------------------------------------------------------------
// Extern globals
// ------------------------------------------------------------
extern SessionInfo  currentSession;
extern SessionState sessionState;
extern unsigned long lastBtRxAt;

// ------------------------------------------------------------
// Function Declarations
// ------------------------------------------------------------
void     btInit();
void     btTick();                        // call every loop
void     btSend(const char* msg);         // send string over BT
void     btSendAck(const char* cmd, const char* status, const char* msg);
void     processCommand(const String& raw);

// Individual command handlers
void     handleHandshake(const String& raw);
void     handleLoadSession(const String& raw);
void     handleActuatorSet(const String& raw);
void     handleTempPadSet(const String& raw);
void     handleStartTherapy(const String& raw);
void     handlePauseTherapy(const String& raw);
void     handleResumeTherapy(const String& raw);
void     handleStopTherapy(const String& raw);
void     handleStatusRequest();
void     handleEmergencyStop();

// Helpers
uint8_t  parseTechnique(const String& t);
uint8_t  parsePadMode(const String& m);
ZoneID   parseZoneID(const String& z);
String   extractField(const String& json, const String& key);
long     extractLong(const String& json, const String& key);
bool     extractBool(const String& json, const String& key);

#endif // BT_HANDLER_H
