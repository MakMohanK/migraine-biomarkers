"""
mouse_listener.py
Standalone Mouse Biomarker Listener
For Migraine Prediction Project

Tracks:
1. session_minutes
2. total_mouse_distance
3. movement_per_minute
4. total_clicks
5. left_clicks
6. right_clicks
7. double_clicks
8. scroll_events
9. avg_mouse_speed
10. idle_time_seconds
11. stop_count
12. migraine_label

OUTPUT:
training_mouse_data.csv

Install:
pip install pynput pandas

Run:
python mouse_listener.py

Stop:
Press ESC on keyboard
"""

import time
import math
import os
import threading
import pandas as pd
from pynput import mouse, keyboard


class MouseBiomarkerListener:

    def __init__(self):

        self.start_time = time.time()

        self.last_position = None
        self.last_move_time = time.time()

        self.total_distance = 0
        self.total_movements = 0

        self.total_clicks = 0
        self.left_clicks = 0
        self.right_clicks = 0
        self.double_clicks = 0

        self.scroll_events = 0

        self.speeds = []
        self.idle_times = []

        self.stop_count = 0

        self.last_click_time = 0

        self.running = True

    # -----------------------------------
    # Mouse Movement
    # -----------------------------------
    def on_move(self, x, y):

        current_time = time.time()

        if self.last_position is not None:

            x1, y1 = self.last_position

            distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

            time_diff = current_time - self.last_move_time

            if time_diff > 0:
                speed = distance / time_diff
                self.speeds.append(speed)

            self.total_distance += distance
            self.total_movements += 1

            # Detect stop
            if distance < 2:
                self.stop_count += 1

            # Detect idle
            if time_diff > 2:
                self.idle_times.append(time_diff)

        self.last_position = (x, y)
        self.last_move_time = current_time

    # -----------------------------------
    # Mouse Clicks
    # -----------------------------------
    def on_click(self, x, y, button, pressed):

        if pressed:

            self.total_clicks += 1

            if button == mouse.Button.left:
                self.left_clicks += 1

            elif button == mouse.Button.right:
                self.right_clicks += 1

            # Double click detection
            now = time.time()

            if now - self.last_click_time < 0.3:
                self.double_clicks += 1

            self.last_click_time = now

    # -----------------------------------
    # Mouse Scroll
    # -----------------------------------
    def on_scroll(self, x, y, dx, dy):
        self.scroll_events += 1

    # -----------------------------------
    # ESC Listener
    # -----------------------------------
    def on_key_release(self, key):

        if key == keyboard.Key.esc:
            self.running = False
            return False

    # -----------------------------------
    # Calculate Metrics
    # -----------------------------------
    def calculate_metrics(self):

        session_minutes = (time.time() - self.start_time) / 60

        movement_per_minute = (
            self.total_movements / session_minutes
            if session_minutes > 0 else 0
        )

        avg_speed = (
            sum(self.speeds) / len(self.speeds)
            if len(self.speeds) > 0 else 0
        )

        avg_idle = (
            sum(self.idle_times) / len(self.idle_times)
            if len(self.idle_times) > 0 else 0
        )

        return {

            "session_minutes": round(session_minutes, 2),

            "total_mouse_distance": round(self.total_distance, 2),

            "movement_per_minute": round(movement_per_minute, 2),

            "total_clicks": self.total_clicks,

            "left_clicks": self.left_clicks,

            "right_clicks": self.right_clicks,

            "double_clicks": self.double_clicks,

            "scroll_events": self.scroll_events,

            "avg_mouse_speed": round(avg_speed, 2),

            "idle_time_seconds": round(avg_idle, 2),

            "stop_count": self.stop_count,

            "migraine_label": 0
        }

    # -----------------------------------
    # Live Dashboard
    # -----------------------------------
    def dashboard(self):

        while self.running:

            metrics = self.calculate_metrics()

            print("\n====== MOUSE BIOMARKER DASHBOARD ======")

            for k, v in metrics.items():
                print(f"{k}: {v}")

            print("Press ESC to stop")

            time.sleep(5)

    # -----------------------------------
    # Save CSV
    # -----------------------------------
    def save_training_data(self):

        filename = "training_mouse_data.csv"

        row = self.calculate_metrics()

        df_new = pd.DataFrame([row])

        if os.path.exists(filename):

            df_old = pd.read_csv(filename)

            df = pd.concat([df_old, df_new], ignore_index=True)

        else:
            df = df_new

        df.to_csv(filename, index=False)

        print(f"\nSaved to {filename}")

    # -----------------------------------
    # Start
    # -----------------------------------
    def start(self):

        print("Mouse Listener Started")
        print("Press ESC to finish session")

        dashboard_thread = threading.Thread(target=self.dashboard)

        dashboard_thread.daemon = True
        dashboard_thread.start()

        mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )

        keyboard_listener = keyboard.Listener(
            on_release=self.on_key_release
        )

        mouse_listener.start()
        keyboard_listener.start()

        keyboard_listener.join()

        mouse_listener.stop()

        self.save_training_data()


# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":

    tracker = MouseBiomarkerListener()
    tracker.start()