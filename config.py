# ─────────────────────────────────────────────────────────────
#  MigraineSense – Global Configuration
# ─────────────────────────────────────────────────────────────

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Database ──────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "data", "migraine_monitor.db")

# ── Sampling intervals (seconds) ─────────────────────────────
KEYBOARD_SAMPLE_INTERVAL   = 5
MOUSE_SAMPLE_INTERVAL      = 5
WEBCAM_SAMPLE_INTERVAL     = 10
SYSTEM_SAMPLE_INTERVAL     = 15
PREDICTION_INTERVAL        = 60          # run predictor every 60 s

# ── Risk thresholds ───────────────────────────────────────────
RISK_LOW       = 30          # 0-29  → safe
RISK_MODERATE  = 55          # 30-54 → watch
RISK_HIGH      = 75          # 55-74 → warning
# ≥ 75 → critical

# ── Webcam ────────────────────────────────────────────────────
WEBCAM_INDEX              = 0
BLINK_THRESHOLD_EAR       = 0.25        # Eye Aspect Ratio
CLOSE_SCREEN_DISTANCE_CM  = 40          # alert if closer than this

# ── Notification cooldown (minutes) ──────────────────────────
NOTIF_COOLDOWN_MINUTES = 15

# ── Dashboard ─────────────────────────────────────────────────
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5050
DASHBOARD_DEBUG = False

# ── Feature-weight tuning (must sum to 1.0) ───────────────────
FEATURE_WEIGHTS = {
    "keyboard": 0.25,
    "mouse":    0.20,
    "webcam":   0.30,
    "system":   0.25,
}

# ── Late-night hours ──────────────────────────────────────────
LATE_NIGHT_START = 23        # 11 PM
LATE_NIGHT_END   = 5         # 5  AM

# ── Session ───────────────────────────────────────────────────
SESSION_IDLE_TIMEOUT = 300   # 5 min of no activity = new session
