"""
camera_listener.py
Standalone Camera Biomarker Listener for Migraine Prediction

Tracks:
1. session_minutes
2. avg_head_tilt
3. head_movement_variability
4. blink_rate_per_min
5. avg_fixation_duration
6. face_distance_score
7. migraine_label

OUTPUT:
camera_training_data.csv

INSTALL:
pip install opencv-python mediapipe pandas numpy

RUN:
python camera_listener.py

STOP:
Press Q
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from math import hypot

# =====================================
# MediaPipe Setup
# =====================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# Landmark IDs
# =====================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_FACE = 234
RIGHT_FACE = 454
NOSE_TIP = 1

# =====================================
# Variables
# =====================================

start_time = time.time()

blink_count = 0
eye_closed = False

head_tilt_list = []
movement_list = []
fixation_list = []
distance_list = []

last_nose_position = None
fixation_start = None

# =====================================
# Eye Aspect Ratio Function
# =====================================

def eye_aspect_ratio(landmarks, eye_points, w, h):

    coords = []

    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        coords.append((x, y))

    vertical_1 = hypot(
        coords[1][0] - coords[5][0],
        coords[1][1] - coords[5][1]
    )

    vertical_2 = hypot(
        coords[2][0] - coords[4][0],
        coords[2][1] - coords[4][1]
    )

    horizontal = hypot(
        coords[0][0] - coords[3][0],
        coords[0][1] - coords[3][1]
    )

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear

# =====================================
# Start Camera
# =====================================

cap = cv2.VideoCapture(0)

print("===================================")
print("Camera Biomarker Listener Started")
print("Press Q to stop")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        # =====================================
        # HEAD TILT
        # =====================================

        left_face = landmarks[LEFT_FACE]
        right_face = landmarks[RIGHT_FACE]

        dx = right_face.x - left_face.x
        dy = right_face.y - left_face.y

        tilt_angle = np.degrees(np.arctan2(dy, dx))

        head_tilt_list.append(abs(tilt_angle))

        # =====================================
        # HEAD MOVEMENT
        # =====================================

        nose = landmarks[NOSE_TIP]

        nose_x = nose.x
        nose_y = nose.y

        if last_nose_position is not None:

            movement = hypot(
                nose_x - last_nose_position[0],
                nose_y - last_nose_position[1]
            )

            movement_list.append(movement)

        last_nose_position = (nose_x, nose_y)

        # =====================================
        # BLINK DETECTION
        # =====================================

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE,
            w,
            h
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE,
            w,
            h
        )

        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < 0.21 and not eye_closed:
            blink_count += 1
            eye_closed = True

        elif avg_ear >= 0.21:
            eye_closed = False

        # =====================================
        # FIXATION DURATION
        # =====================================

        if movement_list:

            recent_move = movement_list[-1]

            if recent_move < 0.002:

                if fixation_start is None:
                    fixation_start = time.time()

            else:

                if fixation_start is not None:
                    fixation_duration = (
                        time.time() - fixation_start
                    )

                    fixation_list.append(
                        fixation_duration
                    )

                    fixation_start = None

        # =====================================
        # FACE DISTANCE SCORE
        # =====================================

        face_width = abs(
            right_face.x - left_face.x
        )

        distance_list.append(face_width)

        # =====================================
        # LIVE DASHBOARD
        # =====================================

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Tilt: {round(tilt_angle,2)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Movement: {round(np.mean(movement_list),5) if movement_list else 0}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "Migraine Camera Biomarker Tracker",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================
# Final Calculations
# =====================================

session_minutes = (
    time.time() - start_time
) / 60

blink_rate = (
    blink_count / session_minutes
    if session_minutes > 0 else 0
)

avg_head_tilt = (
    np.mean(head_tilt_list)
    if head_tilt_list else 0
)

movement_variability = (
    np.std(movement_list)
    if movement_list else 0
)

avg_fixation_duration = (
    np.mean(fixation_list)
    if fixation_list else 0
)

face_distance_score = (
    np.mean(distance_list)
    if distance_list else 0
)

# =====================================
# Final Data Row
# =====================================

data = {
    "session_minutes": round(session_minutes, 2),
    "avg_head_tilt": round(avg_head_tilt, 3),
    "head_movement_variability": round(movement_variability, 5),
    "blink_rate_per_min": round(blink_rate, 2),
    "avg_fixation_duration": round(avg_fixation_duration, 2),
    "face_distance_score": round(face_distance_score, 3),
    "migraine_label": 0
}

# =====================================
# Save CSV
# =====================================

file_name = "camera_training_data.csv"

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

# =====================================
# Final Results
# =====================================

print("\n===================================")
print("SESSION COMPLETE")
print("===================================")

for key, value in data.items():
    print(f"{key}: {value}")

print("\nSaved to:")
print(file_name)

# =====================================
# Cleanup
# =====================================

cap.release()
cv2.destroyAllWindows()