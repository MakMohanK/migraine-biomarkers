# ─────────────────────────────────────────────────────────────
#  Mouse Monitor
#  Tracks: movement speed, jitter, click rate, scroll, idle
# ─────────────────────────────────────────────────────────────

import time
import math
import threading
from collections import deque
from pynput import mouse


class MouseMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False

        self._positions: deque = deque(maxlen=500)   # (x, y, t)
        self._click_times: deque = deque(maxlen=200)
        self._scroll_events: deque = deque(maxlen=200)
        self._total_distance = 0.0
        self._click_count = 0
        self._double_click_count = 0
        self._scroll_count = 0
        self._last_pos = None
        self._last_event_time = 0.0
        self._session_start = 0.0
        self._listener = None

    # ── Public API ────────────────────────────────────────────
    def start(self):
        self._session_start = time.time()
        self._running = True
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._listener.start()

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()

    def get_features(self) -> dict:
        with self._lock:
            now = time.time()
            elapsed = max(now - self._session_start, 1)
            positions = list(self._positions)

            # ── movement speed (px/sec) ───────────────────────
            avg_speed = (self._total_distance / elapsed) if elapsed > 0 else 0

            # ── movement jitter (directional changes per move) ─
            jitter = self._calculate_jitter(positions)

            # ── straight-line efficiency ─────────────────────
            efficiency = self._movement_efficiency(positions)

            # ── click metrics ─────────────────────────────────
            click_rate = (self._click_count / elapsed) * 60   # clicks/min
            double_click_rate = (self._double_click_count / elapsed) * 60

            # ── inter-click interval ──────────────────────────
            clicks = list(self._click_times)
            if len(clicks) > 1:
                icis = [clicks[i+1] - clicks[i] for i in range(len(clicks)-1)]
                mean_ici = (sum(icis) / len(icis)) * 1000   # ms
            else:
                mean_ici = 0

            # ── scroll rate ───────────────────────────────────
            scroll_rate = (self._scroll_count / elapsed) * 60

            # ── idle time ─────────────────────────────────────
            idle_sec = (now - self._last_event_time) if self._last_event_time > 0 else 0

            features = {
                "mouse_avg_speed_px":     round(avg_speed, 2),
                "mouse_jitter":           round(jitter, 4),
                "mouse_efficiency":       round(efficiency, 4),
                "click_rate_per_min":     round(click_rate, 2),
                "double_click_rate":      round(double_click_rate, 2),
                "mean_ici_ms":            round(mean_ici, 2),
                "scroll_rate_per_min":    round(scroll_rate, 2),
                "mouse_idle_sec":         round(idle_sec, 2),
                "total_distance_px":      round(self._total_distance, 1),
            }

            # reset
            self._positions.clear()
            self._click_times.clear()
            self._scroll_events.clear()
            self._total_distance = 0.0
            self._click_count = 0
            self._double_click_count = 0
            self._scroll_count = 0
            self._session_start = now

            return features

    # ── Helpers ───────────────────────────────────────────────
    def _calculate_jitter(self, positions) -> float:
        if len(positions) < 3:
            return 0.0
        direction_changes = 0
        for i in range(1, len(positions) - 1):
            x0, y0, _ = positions[i - 1]
            x1, y1, _ = positions[i]
            x2, y2, _ = positions[i + 1]
            dx1, dy1 = x1 - x0, y1 - y0
            dx2, dy2 = x2 - x1, y2 - y1
            dot = dx1 * dx2 + dy1 * dy2
            mag1 = math.hypot(dx1, dy1)
            mag2 = math.hypot(dx2, dy2)
            if mag1 > 0 and mag2 > 0:
                cos_a = max(-1, min(1, dot / (mag1 * mag2)))
                if cos_a < 0.5:          # direction change > 60°
                    direction_changes += 1
        return direction_changes / max(len(positions) - 2, 1)

    def _movement_efficiency(self, positions) -> float:
        if len(positions) < 2:
            return 1.0
        x0, y0, _ = positions[0]
        xn, yn, _ = positions[-1]
        straight = math.hypot(xn - x0, yn - y0)
        actual = sum(
            math.hypot(positions[i][0] - positions[i-1][0],
                       positions[i][1] - positions[i-1][1])
            for i in range(1, len(positions))
        )
        return (straight / actual) if actual > 0 else 1.0

    # ── Listeners ─────────────────────────────────────────────
    def _on_move(self, x, y):
        if not self._running:
            return
        now = time.time()
        with self._lock:
            if self._last_pos:
                dx = x - self._last_pos[0]
                dy = y - self._last_pos[1]
                self._total_distance += math.hypot(dx, dy)
            self._last_pos = (x, y)
            self._positions.append((x, y, now))
            self._last_event_time = now

    def _on_click(self, x, y, button, pressed):
        if not self._running or not pressed:
            return
        now = time.time()
        with self._lock:
            self._click_count += 1
            self._click_times.append(now)
            self._last_event_time = now
            # detect double click (< 300 ms from last click)
            clicks = list(self._click_times)
            if len(clicks) >= 2 and (clicks[-1] - clicks[-2]) < 0.3:
                self._double_click_count += 1

    def _on_scroll(self, x, y, dx, dy):
        if not self._running:
            return
        now = time.time()
        with self._lock:
            self._scroll_count += 1
            self._scroll_events.append((dx, dy, now))
            self._last_event_time = now
