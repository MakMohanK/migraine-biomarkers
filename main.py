# ─────────────────────────────────────────────────────────────
#  MigraineSense – Main Orchestrator
# ─────────────────────────────────────────────────────────────
# NOTE: eventlet.monkey_patch() has been intentionally REMOVED.
# It replaced threading.Thread with green threads, which broke
# pynput's Windows keyboard hook (it needs a real OS thread).
# Flask-SocketIO now uses async_mode='threading' instead.

import os
import sys
import time
import threading
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PREDICTION_INTERVAL,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    WEBCAM_INDEX,
)

from monitors       import KeyboardMonitor, MouseMonitor, WebcamMonitor, SystemMonitor
from analysis       import FeatureExtractor, RiskCalculator, MigrainePredictor
from storage        import Database
from notifications  import Notifier
from dashboard.app  import push_update, run_dashboard, inject_state, stop_dashboard

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║          🧠  MigraineSense  Early-Warning System         ║
║          Passive · Non-invasive · Real-time               ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
  Dashboard → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}
  Press  Ctrl+C  to stop.
"""

_shutdown = threading.Event()


def _start_in_thread(monitor, label: str):
    """Start a monitor in a real daemon thread (no sleep needed)."""
    def _run():
        try:
            monitor.start()
        except Exception as e:
            print(f"[WARN] {label} failed to start: {e}")
    t = threading.Thread(target=_run, name=label, daemon=True)
    t.start()


class MigraineSense:

    def __init__(self):
        print(BANNER)
        self.db         = Database()
        self.session_id = self.db.start_session()

        self.kb_mon  = KeyboardMonitor()
        self.ms_mon  = MouseMonitor()
        self.wc_mon  = WebcamMonitor(camera_index=WEBCAM_INDEX)
        self.sys_mon = SystemMonitor()

        self.extractor  = FeatureExtractor(
            self.kb_mon, self.ms_mon, self.wc_mon, self.sys_mon
        )
        self.calculator = RiskCalculator()
        self.predictor  = MigrainePredictor()
        self.predictor.load_or_init()

        self.notifier = Notifier(db=self.db)
        self.notifier.register_callback(self._on_alert)

        self._running  = False
        self._max_risk = 0.0

    # ── Lifecycle ─────────────────────────────────────────────
    def start(self):
        self._running = True

        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Starting keyboard monitor …")
        _start_in_thread(self.kb_mon,  "KeyboardMonitor")

        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Starting mouse monitor …")
        _start_in_thread(self.ms_mon,  "MouseMonitor")

        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Starting webcam monitor …")
        _start_in_thread(self.wc_mon,  "WebcamMonitor")

        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Starting system monitor …")
        _start_in_thread(self.sys_mon, "SystemMonitor")

        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Starting feature extractor …")
        self.extractor.start()

        inject_state({}, {}, {}, [], self.db)

        threading.Thread(target=self._warmup,           daemon=True).start()
        threading.Thread(target=self._prediction_loop,  daemon=True).start()
        threading.Thread(target=self._shutdown_watcher, daemon=True).start()

        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Prediction engine active "
              f"(every {PREDICTION_INTERVAL}s) …")
        print(f"{Fore.CYAN}[→]{Style.RESET_ALL} "
              f"Dashboard → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}\n")

        run_dashboard()   # blocks until stop_dashboard() is called

    def stop(self):
        if not self._running:
            return
        print(f"\n{Fore.YELLOW}[!]{Style.RESET_ALL} Shutting down …")
        self._running = False
        for mon in (self.kb_mon, self.ms_mon, self.wc_mon, self.sys_mon):
            try:
                mon.stop()
            except Exception:
                pass
        self.extractor.stop()
        self.predictor.save()
        self.db.end_session(
            self.session_id,
            max_risk=self._max_risk,
            alert_count=self.notifier.get_alert_count(),
        )
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} Session saved. Goodbye!")

    def _shutdown_watcher(self):
        _shutdown.wait()
        self.stop()
        stop_dashboard()
        time.sleep(1)
        os._exit(0)

    def _warmup(self):
        time.sleep(10)
        self._run_prediction()

    def _prediction_loop(self):
        time.sleep(PREDICTION_INTERVAL)
        while self._running:
            self._run_prediction()
            time.sleep(PREDICTION_INTERVAL)

    def _run_prediction(self):
        try:
            features   = self.extractor.get_combined_snapshot() or {}
            risk       = self.calculator.compute(features)
            prediction = self.predictor.predict(features, risk["composite_risk"])
            blended    = prediction["blended_risk"]
            level      = risk["risk_level"]
            eta        = risk.get("eta_minutes")

            self.db.insert_risk(risk, prediction, eta)
            self._max_risk = max(self._max_risk, blended)
            self.notifier.evaluate_and_notify(risk, prediction)
            push_update(risk, prediction, features)

            color = (Fore.RED    if level == "CRITICAL" else
                     Fore.YELLOW if level in ("HIGH", "MODERATE") else Fore.GREEN)
            tsym  = ("↑" if prediction["trend"] > 0.5 else
                     "↓" if prediction["trend"] < -0.5 else "→")
            print(
                f"{color}[{level:8s}]{Style.RESET_ALL} "
                f"Blended:{blended:5.1f}  "
                f"KB:{risk['keyboard_risk']:4.0f}  "
                f"MS:{risk['mouse_risk']:4.0f}  "
                f"CAM:{risk['webcam_risk']:4.0f}  "
                f"SYS:{risk['system_risk']:4.0f}  "
                f"Trend:{tsym}  "
                f"ETA:{str(eta)+'m' if eta else 'N/A':>6}"
            )
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Prediction: {e}")

    def _on_alert(self, alert: dict):
        level = alert.get("level", "")
        msg   = alert.get("message", "")
        color = Fore.RED if level == "CRITICAL" else Fore.YELLOW
        print(f"\n{color}{'─'*58}\n  🔔 [{level}] {msg}\n{'─'*58}{Style.RESET_ALL}\n")


def main():
    app = MigraineSense()

    def _handle(sig, frame):
        _shutdown.set()   # never call sys.exit() here — pynput ctypes hook crash

    signal.signal(signal.SIGINT,  _handle)
    signal.signal(signal.SIGTERM, _handle)
    app.start()


if __name__ == "__main__":
    main()