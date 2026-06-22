# ─────────────────────────────────────────────────────────────
#  Feature Extractor
#  Merges raw monitor outputs into one flat feature vector
# ─────────────────────────────────────────────────────────────

import time
import threading
import psutil
import datetime

from config import KEYBOARD_SAMPLE_INTERVAL, MOUSE_SAMPLE_INTERVAL, \
                   WEBCAM_SAMPLE_INTERVAL, SYSTEM_SAMPLE_INTERVAL

# ── Safe baseline returned on the very first cycle ────────────
def _system_defaults() -> dict:
    """Pull live CPU/RAM/time so the very first push is not all zeros."""
    now  = datetime.datetime.now()
    hour = now.hour
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
    except Exception:
        cpu, mem = 20.0, 50.0
    return {
        # keyboard
        "typing_speed_kpm":  60.0,
        "mean_iki_ms":       200.0,
        "rhythm_cv":         0.3,
        "mean_hold_ms":      80.0,
        "error_rate":        0.02,
        "pause_count":       0,
        "keyboard_idle_sec": 0.0,
        "total_keys":        0,
        # mouse
        "mouse_avg_speed_px":   300.0,
        "mouse_jitter":         0.1,
        "mouse_efficiency":     0.8,
        "click_rate_per_min":   5.0,
        "double_click_rate":    0.5,
        "mean_ici_ms":          500.0,
        "scroll_rate_per_min":  3.0,
        "mouse_idle_sec":       0.0,
        "total_distance_px":    0.0,
        # webcam
        "blink_rate_bpm":    15.0,
        "mean_ear":           0.28,
        "head_tilt_deg":      2.0,
        "head_forward_lean":  0.01,
        "face_proximity_px": 120.0,
        "squint_ratio":       0.05,
        "no_face_ratio":      0.0,
        # system (live readings)
        "avg_cpu_pct":       cpu,
        "avg_mem_pct":       mem,
        "avg_brightness":    70.0,
        "app_switch_rate":   1.0,
        "idle_ratio":        0.1,
        "active_ratio":      0.9,
        "is_late_night":     int(hour >= 23 or hour < 5),
        "is_peak_work":      int(9 <= hour <= 18),
        "hour_of_day":       hour,
        "battery_pct":       100.0,
        "battery_charging":  1,
    }


class FeatureExtractor:
    """
    Polls each monitor on its own cadence and accumulates
    the latest snapshot into a unified feature dict.
    On the very first call, returns live system defaults so the
    dashboard is never empty.
    """

    def __init__(self, keyboard_mon, mouse_mon, webcam_mon, system_mon):
        self._kb  = keyboard_mon
        self._ms  = mouse_mon
        self._wc  = webcam_mon
        self._sys = system_mon

        self._lock    = threading.Lock()
        self._latest  = _system_defaults()   # pre-populate with safe values
        self._history = []
        self._running = False
        self._threads = []

    # ── Public API ────────────────────────────────────────────
    def start(self):
        self._running = True
        schedules = [
            (self._kb,  KEYBOARD_SAMPLE_INTERVAL,  "keyboard"),
            (self._ms,  MOUSE_SAMPLE_INTERVAL,      "mouse"),
            (self._wc,  WEBCAM_SAMPLE_INTERVAL,     "webcam"),
            (self._sys, SYSTEM_SAMPLE_INTERVAL,     "system"),
        ]
        for mon, interval, name in schedules:
            t = threading.Thread(
                target=self._poll_worker,
                args=(mon, interval, name),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False

    def get_latest_features(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def get_history(self, last_n: int = 100) -> list:
        with self._lock:
            return list(self._history[-last_n:])

    def get_combined_snapshot(self) -> dict:
        """Return unified feature dict. Never empty — always has defaults."""
        with self._lock:
            return dict(self._latest)

    # ── Worker ────────────────────────────────────────────────
    def _poll_worker(self, monitor, interval: float, name: str):
        while self._running:
            time.sleep(interval)
            try:
                feats = monitor.get_features()
                if feats:
                    with self._lock:
                        self._latest.update(feats)
                        self._history.append({
                            "timestamp": time.time(),
                            "source":    name,
                            "features":  dict(feats),
                        })
                        if len(self._history) > 1000:
                            self._history = self._history[-1000:]
            except Exception as e:
                print(f"[FeatureExtractor] Error polling {name}: {e}")
