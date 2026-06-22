"""
keyboard_listener.py
FINAL VERSION FOR ML TRAINING

✔ Combines raw + summary into ONE CSV file
✔ One row per session for model training
✔ Press ESC to stop

OUTPUT:
training_keyboard_data.csv

Columns:
session_minutes
total_keys
space_count
backspace_count
keys_per_min
space_per_min
backspace_per_100_chars
avg_pause_seconds
avg_key_hold_seconds
migraine_label   <-- manually fill later

Install:
pip install pynput pandas
"""

import time
import os
import threading
import pandas as pd
from pynput import keyboard


class KeyboardBiomarkerListener:

    def __init__(self):
        self.start_time = time.time()
        self.last_press_time = None

        self.total_keys = 0
        self.space_count = 0
        self.backspace_count = 0

        self.pause_times = []
        self.key_press_time = {}
        self.hold_times = []

        self.running = True

    # --------------------------------
    # Key Press
    # --------------------------------
    def on_press(self, key):
        now = time.time()
        key_name = str(key)

        self.total_keys += 1

        if self.last_press_time is not None:
            pause = now - self.last_press_time
            self.pause_times.append(pause)

        self.last_press_time = now
        self.key_press_time[key_name] = now

        if key == keyboard.Key.space:
            self.space_count += 1

        if key == keyboard.Key.backspace:
            self.backspace_count += 1

    # --------------------------------
    # Key Release
    # --------------------------------
    def on_release(self, key):
        now = time.time()
        key_name = str(key)

        if key_name in self.key_press_time:
            hold = now - self.key_press_time[key_name]
            self.hold_times.append(hold)
            del self.key_press_time[key_name]

        if key == keyboard.Key.esc:
            self.running = False
            return False

    # --------------------------------
    # Calculate Metrics
    # --------------------------------
    def calculate_metrics(self):
        session_minutes = (time.time() - self.start_time) / 60

        keys_per_min = self.total_keys / session_minutes if session_minutes > 0 else 0
        space_per_min = self.space_count / session_minutes if session_minutes > 0 else 0

        backspace_per_100 = (
            (self.backspace_count / self.total_keys) * 100
            if self.total_keys > 0 else 0
        )

        avg_pause = (
            sum(self.pause_times) / len(self.pause_times)
            if len(self.pause_times) > 0 else 0
        )

        avg_hold = (
            sum(self.hold_times) / len(self.hold_times)
            if len(self.hold_times) > 0 else 0
        )

        return {
            "session_minutes": round(session_minutes, 2),
            "total_keys": self.total_keys,
            "space_count": self.space_count,
            "backspace_count": self.backspace_count,
            "keys_per_min": round(keys_per_min, 2),
            "space_per_min": round(space_per_min, 2),
            "backspace_per_100_chars": round(backspace_per_100, 2),
            "avg_pause_seconds": round(avg_pause, 3),
            "avg_key_hold_seconds": round(avg_hold, 3),
            "migraine_label": 0
        }

    # --------------------------------
    # Live Dashboard
    # --------------------------------
    def dashboard(self):
        while self.running:
            m = self.calculate_metrics()

            print("\n====== LIVE SESSION ======")
            for k, v in m.items():
                print(f"{k}: {v}")

            print("Press ESC to stop")
            time.sleep(5)

    # --------------------------------
    # Save To Single CSV
    # --------------------------------
    def save_training_data(self):
        file_name = "training_keyboard_data.csv"
        row = self.calculate_metrics()

        df_new = pd.DataFrame([row])

        if os.path.exists(file_name):
            df_old = pd.read_csv(file_name)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(file_name, index=False)

        print(f"\nSaved to {file_name}")

    # --------------------------------
    # Start Listener
    # --------------------------------
    def start(self):
        print("Keyboard Listener Started")
        print("Press ESC to finish session")

        t = threading.Thread(target=self.dashboard)
        t.daemon = True
        t.start()

        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()

        self.save_training_data()


# --------------------------------
# MAIN
# --------------------------------
if __name__ == "__main__":
    tracker = KeyboardBiomarkerListener()
    tracker.start()