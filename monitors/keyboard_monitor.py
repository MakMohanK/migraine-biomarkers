# ─────────────────────────────────────────────────────────────
#  Keyboard Monitor
#  Tracks: typing speed, pause duration, error rate, key-hold
# ─────────────────────────────────────────────────────────────

import time
import threading
from collections import deque
from pynput import keyboard # type: ignore


class KeyboardMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False

        # raw event buffers
        self._press_times: dict = {}          # key → press timestamp
        self._keystroke_intervals: deque = deque(maxlen=200)
        self._hold_durations: deque = deque(maxlen=200)
        self._total_keys = 0
        self._backspace_count = 0
        self._special_key_count = 0
        self._pause_starts: list = []         # timestamps of pauses > 2 s
        self._last_key_time: float = 0.0
        self._session_start: float = 0.0
        self._listener = None

    # ── Public API ────────────────────────────────────────────
    def start(self):
        self._session_start = time.time()
        self._running = True
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()

    def get_features(self) -> dict:
        """Return aggregated keyboard features and reset rolling buffers."""
        with self._lock:
            now = time.time()
            elapsed = max(now - self._session_start, 1)

            intervals = list(self._keystroke_intervals)
            holds = list(self._hold_durations)

            # typing speed (keys per minute)
            typing_speed = (self._total_keys / elapsed) * 60

            # mean inter-keystroke interval (ms)
            mean_iki = (sum(intervals) / len(intervals) * 1000) if intervals else 0

            # typing rhythm variance (coefficient of variation)
            if len(intervals) > 1:
                mean_i = sum(intervals) / len(intervals)
                variance = sum((x - mean_i) ** 2 for x in intervals) / len(intervals)
                rhythm_cv = (variance ** 0.5 / mean_i) if mean_i > 0 else 0
            else:
                rhythm_cv = 0

            # mean key-hold duration (ms)
            mean_hold = (sum(holds) / len(holds) * 1000) if holds else 0

            # error rate (backspaces / total keys)
            error_rate = (self._backspace_count / self._total_keys) if self._total_keys > 0 else 0

            # pause count (gaps > 3 s within session)
            pause_count = len(self._pause_starts)

            # time since last keystroke (seconds)
            idle_since = (now - self._last_key_time) if self._last_key_time > 0 else 0

            features = {
                "typing_speed_kpm":   round(typing_speed, 2),
                "mean_iki_ms":        round(mean_iki, 2),
                "rhythm_cv":          round(rhythm_cv, 4),
                "mean_hold_ms":       round(mean_hold, 2),
                "error_rate":         round(error_rate, 4),
                "pause_count":        pause_count,
                "keyboard_idle_sec":  round(idle_since, 2),
                "total_keys":         self._total_keys,
            }

            # reset for next window
            self._keystroke_intervals.clear()
            self._hold_durations.clear()
            self._backspace_count = 0
            self._special_key_count = 0
            self._total_keys = 0
            self._pause_starts.clear()
            self._session_start = now

            return features

    # ── Internal callbacks ────────────────────────────────────
    def _on_press(self, key):
        if not self._running:
            return
        now = time.time()
        with self._lock:
            key_id = str(key)
            self._press_times[key_id] = now

            # detect long pause before this keystroke
            if self._last_key_time > 0:
                gap = now - self._last_key_time
                if gap > 3.0:
                    self._pause_starts.append(now)
                else:
                    self._keystroke_intervals.append(gap)

            self._last_key_time = now
            self._total_keys += 1

            # backspace = likely error correction
            if key == keyboard.Key.backspace:
                self._backspace_count += 1

    def _on_release(self, key):
        if not self._running:
            return
        now = time.time()
        with self._lock:
            key_id = str(key)
            press_t = self._press_times.pop(key_id, None)
            if press_t:
                hold = now - press_t
                if hold < 2.0:          # filter accidental long-holds
                    self._hold_durations.append(hold)
