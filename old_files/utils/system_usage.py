"""
UPDATED system_monitor.py
Advanced System Biomarker Listener
for Migraine Prediction Project

NEW FEATURES ADDED:
1. Brightness Level
2. Night Usage Hours
3. Better Break Detection
4. Continuous Screen Time
5. App Switch Rate
6. Session Start Hour
7. Improved ML-ready outputs

Tracks:
- session_minutes
- active_window_switches
- app_switches_per_min
- continuous_screen_time_minutes
- average_cpu_usage
- average_memory_usage
- idle_time_seconds
- estimated_breaks
- screen_brightness_percent
- session_start_hour
- night_usage_flag
- migraine_label

INSTALL:
pip install psutil pandas pygetwindow pyautogui screen-brightness-control

RUN:
python system_monitor.py

STOP:
CTRL + C
"""

import time
import os
import psutil
import pandas as pd
import pygetwindow as gw
import pyautogui
import screen_brightness_control as sbc

from datetime import datetime
from statistics import mean

# =========================================
# VARIABLES
# =========================================

start_time = time.time()

window_history = []
window_switches = 0

cpu_usage_list = []
memory_usage_list = []
brightness_list = []

idle_times = []
break_count = 0

screen_active_seconds = 0

last_mouse_position = pyautogui.position()
last_activity_time = time.time()

IDLE_THRESHOLD = 60

# =========================================
# SESSION TIME FEATURES
# =========================================

session_datetime = datetime.now()

session_start_hour = session_datetime.hour

night_usage_flag = 1 if (
    session_start_hour >= 23
    or session_start_hour <= 4
) else 0

# =========================================
# ACTIVE WINDOW FUNCTION
# =========================================

def get_active_window():

    try:

        window = gw.getActiveWindow()

        if window:
            return window.title

        return "Unknown"

    except:
        return "Unknown"

# =========================================
# MAIN LOOP
# =========================================

print("===================================")
print("ADVANCED SYSTEM BIOMARKER MONITOR")
print("Press CTRL + C to stop")
print("===================================")

try:

    while True:

        current_time = time.time()

        # =========================================
        # ACTIVE WINDOW TRACKING
        # =========================================

        current_window = get_active_window()

        if not window_history:

            window_history.append(current_window)

        elif current_window != window_history[-1]:

            window_switches += 1
            window_history.append(current_window)

        # =========================================
        # CPU + MEMORY
        # =========================================

        cpu = psutil.cpu_percent(interval=1)

        memory = psutil.virtual_memory().percent

        cpu_usage_list.append(cpu)
        memory_usage_list.append(memory)

        # =========================================
        # SCREEN BRIGHTNESS
        # =========================================

        try:

            brightness = sbc.get_brightness()[0]

        except:

            brightness = 50

        brightness_list.append(brightness)

        # =========================================
        # IDLE DETECTION
        # =========================================

        current_mouse_position = pyautogui.position()

        if current_mouse_position != last_mouse_position:

            last_activity_time = current_time
            last_mouse_position = current_mouse_position

        idle_duration = (
            current_time - last_activity_time
        )

        idle_times.append(idle_duration)

        # =========================================
        # BREAK DETECTION
        # =========================================

        if idle_duration > IDLE_THRESHOLD:

            break_count += 1

            last_activity_time = current_time

        # =========================================
        # CONTINUOUS SCREEN TIME
        # =========================================

        if idle_duration < IDLE_THRESHOLD:

            screen_active_seconds += 1

        # =========================================
        # LIVE DASHBOARD
        # =========================================

        elapsed_minutes = (
            current_time - start_time
        ) / 60

        app_switch_rate = (
            window_switches / elapsed_minutes
            if elapsed_minutes > 0 else 0
        )

        os.system("cls" if os.name == "nt" else "clear")

        print("===================================")
        print("SYSTEM BIOMARKER DASHBOARD")
        print("===================================")

        print(f"Session Minutes: {round(elapsed_minutes,2)}")
        print(f"Window Switches: {window_switches}")
        print(f"App Switches/Min: {round(app_switch_rate,2)}")
        print(f"CPU Avg: {round(mean(cpu_usage_list),2)}%")
        print(f"Memory Avg: {round(mean(memory_usage_list),2)}%")
        print(f"Brightness Avg: {round(mean(brightness_list),2)}%")
        print(f"Idle Time: {round(idle_duration,2)} sec")
        print(f"Estimated Breaks: {break_count}")
        print(f"Continuous Screen Time: {screen_active_seconds} sec")
        print(f"Night Usage Flag: {night_usage_flag}")

        time.sleep(1)

# =========================================
# STOP
# =========================================

except KeyboardInterrupt:

    print("\nStopping Monitor...")

# =========================================
# FINAL METRICS
# =========================================

session_minutes = (
    time.time() - start_time
) / 60

app_switches_per_min = (
    window_switches / session_minutes
    if session_minutes > 0 else 0
)

average_cpu_usage = (
    mean(cpu_usage_list)
    if cpu_usage_list else 0
)

average_memory_usage = (
    mean(memory_usage_list)
    if memory_usage_list else 0
)

average_brightness = (
    mean(brightness_list)
    if brightness_list else 0
)

average_idle_time = (
    mean(idle_times)
    if idle_times else 0
)

continuous_screen_time_minutes = (
    screen_active_seconds / 60
)

# =========================================
# FINAL DATA
# =========================================

data = {

    "session_minutes":
        round(session_minutes, 2),

    "active_window_switches":
        window_switches,

    "app_switches_per_min":
        round(app_switches_per_min, 2),

    "continuous_screen_time_minutes":
        round(continuous_screen_time_minutes, 2),

    "average_cpu_usage":
        round(average_cpu_usage, 2),

    "average_memory_usage":
        round(average_memory_usage, 2),

    "idle_time_seconds":
        round(average_idle_time, 2),

    "estimated_breaks":
        break_count,

    "screen_brightness_percent":
        round(average_brightness, 2),

    "session_start_hour":
        session_start_hour,

    "night_usage_flag":
        night_usage_flag,

    "migraine_label":
        0
}

# =========================================
# SAVE CSV
# =========================================

file_name = "system_training_data.csv"

new_df = pd.DataFrame([data])

if os.path.exists(file_name):

    old_df = pd.read_csv(file_name)

    final_df = pd.concat(
        [old_df, new_df],
        ignore_index=True
    )

else:

    final_df = new_df

final_df.to_csv(
    file_name,
    index=False
)

# =========================================
# FINAL RESULTS
# =========================================

print("\n===================================")
print("SESSION COMPLETE")
print("===================================")

for key, value in data.items():

    print(f"{key}: {value}")

print("\nSaved to:")
print(file_name)