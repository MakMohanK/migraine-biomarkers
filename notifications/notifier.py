# ─────────────────────────────────────────────────────────────
#  Notifier
#  Sends desktop + in-app notifications with cooldown logic
# ─────────────────────────────────────────────────────────────

import time
import threading
from config import NOTIF_COOLDOWN_MINUTES

try:
    from plyer import notification as plyer_notif
    PLYER_OK = True
except Exception:
    PLYER_OK = False


LEVEL_META = {
    "LOW":      {"emoji": "🟢", "title": "MigraineSense – All Clear",      "urgency": 0},
    "MODERATE": {"emoji": "🟡", "title": "MigraineSense – Watch Out",       "urgency": 1},
    "HIGH":     {"emoji": "🟠", "title": "MigraineSense – Warning!",        "urgency": 2},
    "CRITICAL": {"emoji": "🔴", "title": "MigraineSense – Take Action Now", "urgency": 3},
}

TIPS = {
    "MODERATE": [
        "Consider taking a 5-minute break.",
        "Look at something 20 ft away for 20 seconds.",
        "Drink a glass of water.",
        "Reduce screen brightness slightly.",
    ],
    "HIGH": [
        "Take a 15-minute break away from the screen.",
        "Dim your display and turn on night mode.",
        "Avoid caffeine and drink water.",
        "Close unnecessary browser tabs to reduce load.",
    ],
    "CRITICAL": [
        "Stop working and rest in a quiet, dark room.",
        "Take any prescribed preventive medication now.",
        "Apply a cold or warm compress to your forehead.",
        "Avoid bright lights and loud sounds.",
    ],
}


class Notifier:
    def __init__(self, db=None):
        self._db             = db
        self._lock           = threading.Lock()
        self._last_sent: dict = {}        # level → last timestamp
        self._tip_index: dict = {}        # level → rotating tip index
        self._alert_count    = 0
        self._callbacks      = []         # UI push callbacks

    # ── Public API ────────────────────────────────────────────
    def register_callback(self, fn):
        """Register a function(alert_dict) for real-time UI pushes."""
        self._callbacks.append(fn)

    def evaluate_and_notify(self, risk: dict, prediction: dict):
        level     = risk.get("risk_level", "LOW")
        score     = prediction.get("blended_risk", 0)
        eta       = risk.get("eta_minutes")

        if level == "LOW":
            return None

        if not self._should_send(level):
            return None

        msg = self._build_message(level, score, eta)
        alert = {
            "level":      level,
            "message":    msg,
            "risk_score": score,
            "timestamp":  time.time(),
        }

        # Desktop notification
        self._desktop_notify(level, msg)

        # Persist
        if self._db:
            self._db.insert_alert(level, msg, score)

        # UI push
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                pass

        with self._lock:
            self._last_sent[level] = time.time()
            self._alert_count += 1

        return alert

    def get_alert_count(self) -> int:
        return self._alert_count

    # ── Helpers ───────────────────────────────────────────────
    def _should_send(self, level: str) -> bool:
        with self._lock:
            last = self._last_sent.get(level, 0)
            cooldown = NOTIF_COOLDOWN_MINUTES * 60
            # CRITICAL has shorter cooldown
            if level == "CRITICAL":
                cooldown = 5 * 60
            return (time.time() - last) >= cooldown

    def _build_message(self, level: str, score: float, eta) -> str:
        meta  = LEVEL_META.get(level, LEVEL_META["MODERATE"])
        emoji = meta["emoji"]

        tips  = TIPS.get(level, TIPS["MODERATE"])
        idx   = self._tip_index.get(level, 0) % len(tips)
        tip   = tips[idx]
        self._tip_index[level] = idx + 1

        if eta:
            eta_str = f"Estimated onset in ~{eta} min. "
        else:
            eta_str = ""

        msg = (
            f"{emoji} Risk score: {score:.0f}/100. "
            f"{eta_str}"
            f"Tip: {tip}"
        )
        return msg

    def _desktop_notify(self, level: str, message: str):
        if not PLYER_OK:
            return
        meta = LEVEL_META.get(level, LEVEL_META["MODERATE"])
        try:
            plyer_notif.notify(
                title=meta["title"],
                message=message,
                app_name="MigraineSense",
                timeout=10,
            )
        except Exception:
            pass
