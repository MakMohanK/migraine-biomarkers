# 🧠 Digital Migraine Therapy Helmet — JSON Specification

## Overview

This folder contains the complete JSON data specification for the **Digital Migraine Therapy Helmet** device.  
The system connects a patient-facing pre-filled form (on laptop/app) to a wearable helmet via **Bluetooth (HC-05)**, and automatically delivers a personalized therapy session based on the user's migraine profile.

---

## System Architecture

```
┌─────────────────────────────────┐
│       Patient Pre-Fills Form    │
│  (Migraine Type, Phase, Pain    │
│   Score, Therapy Preference)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Protocol Matcher Engine      │
│  Looks up therapy_protocol_     │
│  library.json → selects best    │
│  matching PROTO-XXX             │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Session Builder               │
│  Generates sample_therapy_      │
│  session.json payload           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Bluetooth Transmitter         │
│   HC-05 (UART Serial 9600 baud) │
│   Sends CMD packets from        │
│   bluetooth_command_packet.json │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Digital Helmet Device       │
│                                 │
│  5 Actuators (Pressure Zones)   │
│  ┌──────────────────────────┐   │
│  │ • Frontal Lobe           │   │
│  │ • Parietal Lobe          │   │
│  │ • Occipital Lobe         │   │
│  │ • Right Temporal (ear)   │   │
│  │ • Left Temporal (ear)    │   │
│  └──────────────────────────┘   │
│                                 │
│  2 Temperature Pads             │
│  ┌──────────────────────────┐   │
│  │ • Frontal Pad            │   │
│  │ • Occipital Pad          │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

---

## Files in This Folder

| File | Purpose |
|---|---|
| `therapy_schema.json` | Full JSON Schema (validator) for any therapy session payload |
| `sample_therapy_session.json` | A real filled example session for **Migraine With Aura — Pain Score 7** |
| `therapy_protocol_library.json` | Pre-built protocols for all migraine types, phases & pain levels |
| `bluetooth_command_packet.json` | Bluetooth HC-05 command packet spec + full example send sequence |
| `README.md` | This documentation file |

---

## Device Zones & Their Roles

### 🔵 Actuators (Pressure / Vibration / Massage)

| Zone ID | Physical Location | Primary Therapeutic Role |
|---|---|---|
| `frontal_lobe` | Forehead band | Reduces vascular pressure, tension relief |
| `parietal_lobe` | Top of skull | Improves scalp circulation, general tension |
| `occipital_lobe` | Back of head | Sub-occipital trigger point release |
| `right_temporal` | Near right ear | Right-side temporal artery pressure |
| `left_temporal` | Near left ear | Left-side temporal artery pressure |

### 🌡️ Temperature Pads (Heat / Cold / Alternating)

| Pad ID | Physical Location | Primary Therapeutic Role |
|---|---|---|
| `frontal_pad` | Forehead | Cold = reduce inflammation & numb pain receptors |
| `occipital_pad` | Back of head | Heat = relax muscles; Alternating = boost blood flow |

---

## Massage Techniques Reference

| Technique | Description | Best Used For |
|---|---|---|
| `SUSTAINED_PRESSURE` | Constant even pressure held over area | Frontal vascular compression |
| `CIRCULAR` | Circular motion pattern at set frequency | Circulation improvement |
| `RHYTHMIC_TAPPING` | Rhythmic on/off tapping at set Hz | Temporal artery zones |
| `VIBRATION` | High-frequency vibration | Cluster headache deep relief |
| `PETRISSAGE` | Kneading / compression-release | Muscle tension in parietal zone |
| `EFFLEURAGE` | Very light gliding strokes | Gentle onset, postdrome, vestibular |
| `TRIGGER_POINT` | Sustained deep pressure on trigger point | Occipital sub-cranial muscles |

---

## Therapy Protocol Library Summary

| Protocol ID | Migraine Type | Phase | Pain Range | Profile | Duration |
|---|---|---|---|---|---|
| PROTO-001 | Tension | Headache | 1–4 | Gentle | 20 min |
| PROTO-002 | Tension | Headache | 5–10 | Moderate | 30 min |
| PROTO-003 | Migraine With Aura | Aura | 1–5 | Gentle | 15 min |
| PROTO-004 | Migraine With Aura | Headache | 6–10 | Intensive | 35 min |
| PROTO-005 | Cluster | Headache | 7–10 | Intensive | 30 min |
| PROTO-006 | Migraine Without Aura | Headache | 3–6 | Moderate | 25 min |
| PROTO-007 | Vestibular | Headache | 1–10 | Gentle | 20 min |
| PROTO-008 | Hemiplegic | Headache | 1–10 | Gentle (Caution) | 15 min |
| PROTO-009 | Any | Prodrome | 1–4 | Gentle | 15 min |
| PROTO-010 | Any | Postdrome | 1–3 | Gentle | 20 min |

---

## Bluetooth Communication Flow

```
HOST (Laptop/App)                     DEVICE (Helmet)
       │                                     │
       │──── CMD_HANDSHAKE ─────────────────▶│
       │◀─── ACK + device_id + battery ──────│
       │                                     │
       │──── CMD_LOAD_SESSION ──────────────▶│
       │◀─── ACK ────────────────────────────│
       │                                     │
       │──── CMD_ACTUATOR_SET (×5 zones) ───▶│
       │◀─── ACK (each) ─────────────────────│
       │                                     │
       │──── CMD_TEMP_PAD_SET (×2 pads) ────▶│
       │◀─── ACK (each) ─────────────────────│
       │                                     │
       │──── CMD_START_THERAPY ─────────────▶│
       │◀─── ACK + started_at ───────────────│
       │                                     │
       │  [Therapy Running — polling loop]   │
       │──── CMD_STATUS_REQUEST ────────────▶│
       │◀─── status + zone readings ─────────│
       │                                     │
       │──── CMD_STOP_THERAPY ──────────────▶│
       │◀─── ACK + total_elapsed_seconds ────│
```

---

## Safety Limits

| Parameter | Limit |
|---|---|
| Max actuator pressure | 30 kPa |
| Max pad temperature | 42 °C |
| Min pad temperature | 10 °C |
| Auto-stop if BT signal lost | ✅ Yes |
| Emergency stop command | `CMD_EMERGENCY_STOP` → code `ESTOP::0xFF` |
| User pause allowed | ✅ Yes |

---

## Integration with Digital Biomarker (Prediction Phase)

The **Digital Biomarker Algorithm** (migraine arrival prediction) feeds directly into the pre-populated form:

```
Biomarker Algorithm Output
    │
    ├── predicted_migraine_type  ──▶ maps to protocol migraine_type
    ├── predicted_phase          ──▶ maps to protocol migraine_phase
    ├── estimated_pain_score     ──▶ maps to protocol pain_score_range
    └── recommended_profile      ──▶ maps to therapy_profile
```

This closes the loop between **prediction (digital biomarker)** and **treatment (helmet therapy)**.
