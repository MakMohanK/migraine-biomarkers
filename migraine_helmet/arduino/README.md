# 🧠 Migraine Therapy Helmet — Arduino Firmware

## Board Recommendation
| Board | Status | Notes |
|---|---|---|
| **Arduino Mega 2560** | ✅ Recommended | Uses hardware `Serial1` for HC-05 — most reliable |
| Arduino Uno | ⚠️ Supported | Uses `SoftwareSerial` on pins 10 (RX), 11 (TX) |

> Set `USE_MEGA` in `config.h` to `true` (Mega) or `false` (Uno)

---

## 📁 File Structure

```
arduino/
└── migraine_helmet/
    ├── migraine_helmet.ino   ← Main entry point (setup + loop)
    ├── config.h              ← Pin map, relay logic, safety limits
    ├── relay_controller.h    ← Actuator & temp pad structs + declarations
    ├── relay_controller.cpp  ← Relay ON/OFF duty cycle logic
    ├── bt_handler.h          ← BT command declarations & session struct
    └── bt_handler.cpp        ← Full command parser & handler logic
```

---

## 🔌 Wiring Diagram

### Actuator Relays (5 relays — pressure actuators)

```
Arduino Pin   Relay Module   Helmet Zone
─────────────────────────────────────────────────
Pin 2      →  Relay  1   →  Frontal Lobe Actuator
Pin 3      →  Relay  2   →  Parietal Lobe Actuator
Pin 4      →  Relay  3   →  Occipital Lobe Actuator
Pin 5      →  Relay  4   →  Right Temporal (near right ear)
Pin 6      →  Relay  5   →  Left Temporal  (near left ear)
```

### Temperature Pad Relays (6 relays — 3 per pad)

```
Arduino Pin   Relay Module   Function
─────────────────────────────────────────────────────────
Pin 7      →  Relay  6   →  Frontal Pad   — HEAT element
Pin 8      →  Relay  7   →  Frontal Pad   — COLD element (Peltier)
Pin 9      →  Relay  8   →  Frontal Pad   — MASTER power switch
Pin 10     →  Relay  9   →  Occipital Pad — HEAT element
Pin 11     →  Relay 10   →  Occipital Pad — COLD element (Peltier)
Pin 12     →  Relay 11   →  Occipital Pad — MASTER power switch
```

> **Why 3 relays per pad?**
> - `MASTER` relay = main power safety switch (cuts everything)
> - `HEAT` relay = enables heating element
> - `COLD` relay = enables Peltier cooler
> - Only ONE of HEAT/COLD is ever ON at a time — master ensures safety

### HC-05 Bluetooth Wiring (Mega)

```
HC-05 Pin   →   Arduino Mega
─────────────────────────────────────────────
VCC         →   5V
GND         →   GND
TX          →   RX1 (Pin 19)
RX          →   TX1 (Pin 18) via voltage divider*
```

> ⚠️ **Voltage divider required on RX line:**
> HC-05 RX accepts 3.3V max. Use:
> `Arduino TX1 → 1kΩ → HC-05 RX`
> `Junction → 2kΩ → GND`

### HC-05 Bluetooth Wiring (Uno / SoftwareSerial)

```
HC-05 TX  →  Arduino Pin 10 (SoftwareSerial RX)
HC-05 RX  →  Arduino Pin 11 (SoftwareSerial TX) via voltage divider
```

---

## ⚡ Full Circuit Diagram (Text)

```
                    +5V
                     │
           ┌─────────┴──────────┐
           │   Arduino Mega     │
           │                    │
    Pin 2──┤──►[Relay 1]──► Frontal Actuator
    Pin 3──┤──►[Relay 2]──► Parietal Actuator
    Pin 4──┤──►[Relay 3]──► Occipital Actuator
    Pin 5──┤──►[Relay 4]──► Right Temporal Actuator
    Pin 6──┤──►[Relay 5]──► Left Temporal Actuator
           │
    Pin 7──┤──►[Relay 6]──► Frontal Pad HEAT
    Pin 8──┤──►[Relay 7]──► Frontal Pad COLD
    Pin 9──┤──►[Relay 8]──► Frontal Pad MASTER
           │
    Pin10──┤──►[Relay 9] ──► Occipital Pad HEAT
    Pin11──┤──►[Relay 10]──► Occipital Pad COLD
    Pin12──┤──►[Relay 11]──► Occipital Pad MASTER
           │
    Pin13──┤──► Onboard LED (heartbeat)
           │
   RX1(19)─┤◄── HC-05 TX
   TX1(18)─┤──► HC-05 RX (via 1kΩ/2kΩ divider)
           └────────────────────┘
```

---

## 🧠 Firmware Architecture

```
loop()
  │
  ├── btTick()                  ← Read HC-05 serial bytes
  │     └── processCommand()    ← Route to correct handler
  │           ├── handleHandshake()
  │           ├── handleLoadSession()
  │           ├── handleActuatorSet()
  │           ├── handleTempPadSet()
  │           ├── handleStartTherapy()
  │           ├── handlePauseTherapy()
  │           ├── handleResumeTherapy()
  │           ├── handleStopTherapy()
  │           ├── handleStatusRequest()
  │           └── handleEmergencyStop()
  │
  ├── tickActuators()           ← ON/OFF duty cycle per actuator zone
  ├── tickTempPads()            ← HEAT/COLD/ALTERNATING per temp pad
  ├── checkSessionCompletion()  ← Auto-stop when total duration elapsed
  ├── checkAllZonesFinished()   ← Auto-stop if all zones done early
  └── heartbeatTick()           ← LED blink (fast=running, slow=idle)
```

---

## 📡 Bluetooth Command Reference

All commands are JSON strings sent over HC-05 serial at **9600 baud**, terminated with `\r\n`.

| Command | Code | Description |
|---|---|---|
| `CMD_HANDSHAKE` | 0x01 | Connect & verify device |
| `CMD_LOAD_SESSION` | 0x02 | Load session metadata |
| `CMD_ACTUATOR_SET` | 0x03 | Configure one/all actuator zones |
| `CMD_TEMP_PAD_SET` | 0x04 | Configure a temperature pad |
| `CMD_START_THERAPY` | 0x05 | Start the loaded session |
| `CMD_PAUSE_THERAPY` | 0x06 | Pause active session |
| `CMD_RESUME_THERAPY` | 0x07 | Resume paused session |
| `CMD_STOP_THERAPY` | 0x08 | Gracefully stop & reset |
| `CMD_STATUS_REQUEST` | 0x09 | Get real-time device status |
| `CMD_EMERGENCY_STOP` | 0xFF | Instantly cut all power |

---

## 🔄 Session State Machine

```
           ┌──────────┐
           │   IDLE   │◄──────────────────────────────┐
           └────┬─────┘                               │
                │ CMD_LOAD_SESSION                     │
                ▼                                     │
           ┌──────────┐                               │
           │  LOADED  │                               │
           └────┬─────┘                               │
                │ CMD_START_THERAPY                    │
                ▼                                     │
           ┌──────────┐   CMD_PAUSE_THERAPY  ┌──────────────┐
           │ RUNNING  │─────────────────────►│   PAUSED     │
           │          │◄────────────────────-│              │
           └────┬─────┘  CMD_RESUME_THERAPY  └──────────────┘
                │
                │ duration elapsed / CMD_STOP / CMD_EMERGENCY_STOP
                ▼
           ┌──────────┐
           │ FINISHED │──────────────────────────────►IDLE
           └──────────┘
```

---

## ⚙️ Intensity → Duty Cycle Map

| Intensity Level | Relay ON Time | Description |
|---|---|---|
| 0 | 0 ms | OFF |
| 1 | 200 ms | Very Light |
| 2 | 400 ms | Light |
| 3 | 600 ms | Mild |
| 4 | 800 ms | Moderate-Light |
| 5 | 1000 ms | Medium |
| 6 | 1200 ms | Moderate-Strong |
| 7 | 1500 ms | Strong |
| 8 | 1800 ms | Very Strong |
| 9 | 2200 ms | Intense |
| 10 | 3000 ms | Maximum |

> OFF time is fixed at `500ms` between each pulse.

---

## 🛡️ Safety Features

| Feature | Behaviour |
|---|---|
| BT signal loss watchdog | Pauses all zones after 5 sec no signal |
| Emergency stop | `CMD_EMERGENCY_STOP` cuts all relays instantly (no ramp) |
| Session hard cap | Max 60 min session — auto-stops regardless |
| Relay mutual exclusion | HEAT and COLD relays on same pad can never be ON together |
| MASTER pad relay | Cuts all pad power before switching HEAT/COLD |
| Voltage divider | Protects HC-05 RX pin from 5V Arduino TX signal |

---

## 🚀 How to Upload

1. Open `migraine_helmet/migraine_helmet.ino` in **Arduino IDE**
2. Set board: `Tools → Board → Arduino Mega 2560`
3. Set port: `Tools → Port → COMx` (your Arduino port)
4. Click **Upload**
5. Open Serial Monitor at `115200 baud` to see debug output
6. Pair HC-05 on host laptop (default PIN: `1234`)
7. Send therapy JSON commands from your host application

---

## 📦 Dependencies

- No external libraries required for **Mega**
- For **Uno** only: `SoftwareSerial` (built into Arduino IDE)
