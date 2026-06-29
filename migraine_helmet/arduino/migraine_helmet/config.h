// ============================================================
//  config.h — Pin Mapping & Global Configuration
//  Digital Migraine Therapy Helmet
//  Arduino Mega 2560 (recommended) or Uno
// ============================================================

#ifndef CONFIG_H
#define CONFIG_H

// ------------------------------------------------------------
// BLUETOOTH (HC-05) — Hardware Serial1 on Mega
// On Uno: use SoftwareSerial on pins 10(RX), 11(TX)
// ------------------------------------------------------------
#define BT_BAUD         9600
#define USE_MEGA        true        // set false if using Uno

// ------------------------------------------------------------
// ACTUATOR RELAY PINS  (5 pressure actuators)
// Relay module: LOW = ON, HIGH = OFF (active-low relay board)
// ------------------------------------------------------------
#define RELAY_FRONTAL       2     // Frontal Lobe Actuator
#define RELAY_PARIETAL      3     // Parietal Lobe Actuator
#define RELAY_OCCIPITAL     4     // Occipital Lobe Actuator
#define RELAY_RIGHT_TEMP    5     // Right Temporal (near right ear)
#define RELAY_LEFT_TEMP     6     // Left Temporal (near left ear)

// ------------------------------------------------------------
// TEMPERATURE PAD RELAY PINS  (2 pads × 3 relays each)
//   Each pad has:
//     HEAT relay  — energises heating element
//     COLD relay  — energises peltier/cold element
//     COMMON relay— main power to the pad (safety master)
// ------------------------------------------------------------
// Frontal Pad
#define RELAY_FRONTAL_PAD_HEAT      7
#define RELAY_FRONTAL_PAD_COLD      8
#define RELAY_FRONTAL_PAD_MASTER    9

// Occipital Pad
#define RELAY_OCCIPITAL_PAD_HEAT    10
#define RELAY_OCCIPITAL_PAD_COLD    11
#define RELAY_OCCIPITAL_PAD_MASTER  12

// ------------------------------------------------------------
// STATUS LED
// ------------------------------------------------------------
#define LED_STATUS      13        // Built-in LED = system heartbeat

// ------------------------------------------------------------
// RELAY LOGIC (active-low relay boards)
// ------------------------------------------------------------
#define RELAY_ON        LOW
#define RELAY_OFF       HIGH

// ------------------------------------------------------------
// SAFETY LIMITS
// ------------------------------------------------------------
#define MAX_SESSION_DURATION_MS   3600000UL  // 60 min hard cap
#define BT_TIMEOUT_MS             5000UL     // 5 sec no signal → pause
#define CYCLE_TICK_MS             500UL      // main loop tick rate
#define ESTOP_CODE                "ESTOP::0xFF"

// ------------------------------------------------------------
// THERAPY PROFILES  (intensity → on/off duty cycle in ms)
// Maps intensity_level (1–10) to ON time in ms
// OFF time is always CYCLE_TICK_MS
// ------------------------------------------------------------
// Duty cycle table: index = intensity_level (0..10)
const unsigned long INTENSITY_ON_MS[11] = {
  0,      // 0 = OFF
  200,    // 1 = very light
  400,    // 2
  600,    // 3
  800,    // 4
  1000,   // 5 = medium
  1200,   // 6
  1500,   // 7
  1800,   // 8
  2200,   // 9
  3000    // 10 = maximum
};

// Temperature pad mode codes
#define PAD_OFF         0
#define PAD_HEAT        1
#define PAD_COLD        2
#define PAD_ALTERNATING 3

#endif // CONFIG_H
