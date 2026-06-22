# ─────────────────────────────────────────────────────────────
#  Webcam Monitor
#  Tracks: blink rate, head tilt, face proximity, eye strain
# ─────────────────────────────────────────────────────────────

import time
import math
import threading
import numpy as np

try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# MediaPipe landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
NOSE_TIP  = 1
CHIN      = 152
LEFT_EAR  = 234
RIGHT_EAR = 454

# Face mesh connections for head pose
FACE_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]


class WebcamMonitor:
    def __init__(self, camera_index: int = 0):
        self._cam_index  = camera_index
        self._lock       = threading.Lock()
        self._running    = False
        self._thread     = None

        # rolling counters
        self._blink_count        = 0
        self._blink_timestamps   = []
        self._ear_values         = []          # Eye Aspect Ratio samples
        self._head_tilt_values   = []          # degrees
        self._head_forward_vals  = []          # forward lean proxy
        self._face_proximity     = []          # face bbox height (px)
        self._squint_events      = 0
        self._no_face_seconds    = 0
        self._frame_count        = 0
        self._session_start      = 0.0

        self._eyes_closed        = False
        self._ear_threshold      = 0.25

    # ── Public API ────────────────────────────────────────────
    def start(self):
        if not MEDIAPIPE_AVAILABLE:
            print("[WebcamMonitor] mediapipe/cv2 not available – skipping webcam.")
            return
        self._session_start = time.time()
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_features(self) -> dict:
        with self._lock:
            now     = time.time()
            elapsed = max(now - self._session_start, 1)

            # blink rate (blinks per minute)
            blink_rate = (self._blink_count / elapsed) * 60

            # mean EAR (lower = more closed / strained)
            mean_ear = float(np.mean(self._ear_values)) if self._ear_values else 0.28

            # head tilt (mean absolute degrees from vertical)
            mean_tilt = float(np.mean([abs(v) for v in self._head_tilt_values])) \
                        if self._head_tilt_values else 0.0

            # head forward lean (mean)
            mean_fwd = float(np.mean(self._head_forward_vals)) \
                       if self._head_forward_vals else 0.0

            # face proximity (larger bbox = closer to screen)
            mean_prox = float(np.mean(self._face_proximity)) \
                        if self._face_proximity else 0.0

            # squint ratio
            squint_ratio = self._squint_events / max(self._frame_count, 1)

            # no-face ratio
            no_face_ratio = self._no_face_seconds / elapsed

            features = {
                "blink_rate_bpm":     round(blink_rate,   2),
                "mean_ear":           round(mean_ear,      4),
                "head_tilt_deg":      round(mean_tilt,     2),
                "head_forward_lean":  round(mean_fwd,      4),
                "face_proximity_px":  round(mean_prox,     1),
                "squint_ratio":       round(squint_ratio,  4),
                "no_face_ratio":      round(no_face_ratio, 4),
            }

            # reset
            self._blink_count       = 0
            self._blink_timestamps  = []
            self._ear_values        = []
            self._head_tilt_values  = []
            self._head_forward_vals = []
            self._face_proximity    = []
            self._squint_events     = 0
            self._no_face_seconds   = 0
            self._frame_count       = 0
            self._session_start     = now

            return features

    # ── Background capture loop ───────────────────────────────
    def _capture_loop(self):
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        cap = cv2.VideoCapture(self._cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,           10)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            h, w   = frame.shape[:2]

            with self._lock:
                self._frame_count += 1

                if result.multi_face_landmarks:
                    lms = result.multi_face_landmarks[0].landmark
                    self._process_eyes(lms, w, h)
                    self._process_head_pose(lms, w, h)
                    self._process_proximity(lms, w, h)
                else:
                    self._no_face_seconds += 0.1

            time.sleep(0.1)   # ~10 fps is enough

        cap.release()
        face_mesh.close()

    # ── Eye processing ────────────────────────────────────────
    def _process_eyes(self, lms, w, h):
        left_ear  = self._eye_aspect_ratio(lms, LEFT_EYE,  w, h)
        right_ear = self._eye_aspect_ratio(lms, RIGHT_EYE, w, h)
        ear       = (left_ear + right_ear) / 2.0
        self._ear_values.append(ear)

        if ear < 0.20:
            self._squint_events += 1

        if ear < self._ear_threshold:
            if not self._eyes_closed:
                self._eyes_closed = True
        else:
            if self._eyes_closed:
                self._blink_count += 1
                self._blink_timestamps.append(time.time())
                self._eyes_closed = False

    def _eye_aspect_ratio(self, lms, indices, w, h):
        pts = [(lms[i].x * w, lms[i].y * h) for i in indices]
        A = math.dist(pts[1], pts[5])
        B = math.dist(pts[2], pts[4])
        C = math.dist(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0

    # ── Head pose ─────────────────────────────────────────────
    def _process_head_pose(self, lms, w, h):
        # Tilt: angle between left-ear → right-ear vector and horizontal
        le = (lms[LEFT_EAR].x  * w, lms[LEFT_EAR].y  * h)
        re = (lms[RIGHT_EAR].x * w, lms[RIGHT_EAR].y * h)
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        tilt_deg = math.degrees(math.atan2(dy, dx))
        self._head_tilt_values.append(tilt_deg)

        # Forward lean: nose tip y relative to chin y (normalised)
        nose  = lms[NOSE_TIP].z
        chin  = lms[CHIN].z
        self._head_forward_vals.append(abs(nose - chin))

    # ── Face proximity ────────────────────────────────────────
    def _process_proximity(self, lms, w, h):
        nose_y  = lms[NOSE_TIP].y  * h
        chin_y  = lms[CHIN].y      * h
        face_h  = abs(chin_y - nose_y)
        self._face_proximity.append(face_h)
