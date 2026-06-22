# ─────────────────────────────────────────────────────────────
#  System Monitor
#  Tracks: CPU, memory, brightness, app-switches, idle, time
# ─────────────────────────────────────────────────────────────

import time
import threading
import datetime
import psutil

try:
    import screen_brightness_control as sbc
    SBC_AVAILABLE = True
except Exception:
    SBC_AVAILABLE = False

try:
    import pygetwindow as gw
    GW_AVAILABLE = True
except Exception:
    GW_AVAILABLE = False


class SystemMonitor:
    def __init__(self):
        self._lock            = threading.Lock()
        self._running         = False
        self._thread          = None

        self._cpu_samples     = []
        self._mem_samples     = []
        self._brightness_samples = []
        self._app_switches    = 0
        self._last_window     = None
        self._idle_seconds    = 0
        self._active_seconds  = 0
        self._session_start   = 0.0
        self._sample_interval = 5          # seconds between polls

    # ── Public API ────────────────────────────────────────────
    def start(self):
        self._session_start = time.time()
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_features(self) -> dict:
        with self._lock:
            now     = time.time()
            elapsed = max(now - self._session_start, 1)

            avg_cpu = (sum(self._cpu_samples) / len(self._cpu_samples)
                       if self._cpu_samples else 0.0)
            avg_mem = (sum(self._mem_samples) / len(self._mem_samples)
                       if self._mem_samples else 0.0)
            avg_bright = (sum(self._brightness_samples) / len(self._brightness_samples)
                          if self._brightness_samples else 50.0)

            app_switch_rate = (self._app_switches / elapsed) * 60   # per min

            idle_ratio   = self._idle_seconds   / elapsed
            active_ratio = self._active_seconds / elapsed

            now_dt        = datetime.datetime.now()
            hour          = now_dt.hour
            is_late_night = int(hour >= 23 or hour < 5)
            is_peak_work  = int(9 <= hour <= 18)

            # battery (if laptop)
            battery_pct = 100.0
            battery_charging = 1
            try:
                bat = psutil.sensors_battery()
                if bat:
                    battery_pct     = bat.percent
                    battery_charging = int(bat.power_plugged)
            except Exception:
                pass

            features = {
                "avg_cpu_pct":        round(avg_cpu,          2),
                "avg_mem_pct":        round(avg_mem,          2),
                "avg_brightness":     round(avg_bright,       2),
                "app_switch_rate":    round(app_switch_rate,  4),
                "idle_ratio":         round(idle_ratio,       4),
                "active_ratio":       round(active_ratio,     4),
                "is_late_night":      is_late_night,
                "is_peak_work":       is_peak_work,
                "hour_of_day":        hour,
                "battery_pct":        round(battery_pct,      1),
                "battery_charging":   battery_charging,
            }

            # reset
            self._cpu_samples        = []
            self._mem_samples        = []
            self._brightness_samples = []
            self._app_switches       = 0
            self._idle_seconds       = 0
            self._active_seconds     = 0
            self._session_start      = now

            return features

    # ── Background poll loop ──────────────────────────────────
    def _poll_loop(self):
        while self._running:
            try:
                with self._lock:
                    # CPU & memory
                    self._cpu_samples.append(psutil.cpu_percent(interval=None))
                    self._mem_samples.append(psutil.virtual_memory().percent)

                    # Screen brightness
                    if SBC_AVAILABLE:
                        try:
                            b = sbc.get_brightness()
                            val = b[0] if isinstance(b, list) else b
                            self._brightness_samples.append(float(val))
                        except Exception:
                            self._brightness_samples.append(50.0)
                    else:
                        self._brightness_samples.append(50.0)

                    # Active window / app switches
                    if GW_AVAILABLE:
                        try:
                            wins = gw.getActiveWindow()
                            title = wins.title if wins else ""
                            if title and title != self._last_window:
                                self._app_switches += 1
                                self._last_window   = title
                        except Exception:
                            pass

                    # Idle vs active (rough: CPU < 5% → idle)
                    cpu_now = self._cpu_samples[-1] if self._cpu_samples else 0
                    if cpu_now < 5.0:
                        self._idle_seconds   += self._sample_interval
                    else:
                        self._active_seconds += self._sample_interval

            except Exception as e:
                pass

            time.sleep(self._sample_interval)
