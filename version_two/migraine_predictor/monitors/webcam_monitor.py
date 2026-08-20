"""
Webcam/Facial Monitoring Module
Tracks facial cues to detect eye strain, fatigue, and tension indicators.

Updated for MediaPipe 1.0 Tasks API (mp.solutions removed in 1.0).

Features tracked:
- Blink rate and duration
- Head tilt/position
- Face distance from screen
- Eye openness
- Facial movement/fidgeting
"""

import threading
import time
import os
import math
from datetime import datetime
from collections import deque
import statistics

# ── MediaPipe 1.0 Tasks API import ──────────────────────────────────────────
OPENCV_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not available. Webcam monitoring will use simulation mode.")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("Warning: MediaPipe not available. Webcam monitoring will use simulation mode.")

# Path to the face landmarker model (downloaded once)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, '..', 'models', 'face_landmarker.task')
_MODEL_PATH = os.path.normpath(_MODEL_PATH)


class WebcamMonitor:
    """Monitors webcam for facial cues indicating fatigue and strain"""
    
    def __init__(self, window_size=60, camera_index=0):
        """
        Initialize webcam monitor.
        Args:
            window_size: Time window in seconds for calculating features
            camera_index: Camera device index (default 0)
        """
        self.window_size  = window_size
        self.camera_index = camera_index
        self._running     = False
        self._thread      = None
        self._cap         = None
        self._landmarker  = None  # FaceLandmarker instance

        # ── Data buffers ──────────────────────────────────────────────────
        self.blink_times     = deque(maxlen=200)
        self.blink_durations = deque(maxlen=200)
        self.head_positions  = deque(maxlen=500)
        self.face_distances  = deque(maxlen=500)
        self.eye_openness    = deque(maxlen=500)
        self.face_movements  = deque(maxlen=300)
        self.timestamps      = deque(maxlen=500)

        # ── State ─────────────────────────────────────────────────────────
        self._eye_closed         = False
        self._eye_close_start    = None
        self._last_face_position = None
        self._frames_processed   = 0
        self._blink_count        = 0

        # MediaPipe Face Mesh landmark indices (478 landmarks in 1.0)
        self.LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]
        self.baseline = {
            'blink_rate':          15,
            'avg_eye_openness':    0.3,
            'face_distance':       50,
            'head_tilt_threshold': 15,
        }
    
    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start webcam monitoring"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[OK] Webcam monitor started")
    
    def stop(self):
        """Stop webcam monitoring"""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        print("[OK] Webcam monitor stopped")
    
    def is_running(self):
        """Check if monitor is running"""
        return self._running
    
    # ── Main loop ─────────────────────────────────────────────────────────
    def _monitor_loop(self):
        """Main monitoring loop"""
        if OPENCV_AVAILABLE and MEDIAPIPE_AVAILABLE and os.path.exists(_MODEL_PATH):
            self._opencv_monitor_loop()
        else:
            if not os.path.exists(_MODEL_PATH):
                print(f"Warning: Model file not found at {_MODEL_PATH}. Using simulation mode.")
            self._simulation_loop()
    
    def _build_landmarker(self):
        """Create a MediaPipe 1.0 FaceLandmarker in IMAGE running mode."""
        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options   = mp_vision.FaceLandmarkerOptions(
            base_options                  = base_opts,
            running_mode                  = mp_vision.RunningMode.IMAGE,
            num_faces                     = 1,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    def _opencv_monitor_loop(self):
        """Monitor using OpenCV + MediaPipe 1.0 tasks API."""
        try:
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                print("Warning: Could not open camera. Switching to simulation mode.")
                self._simulation_loop()
                return
            
            self._landmarker = self._build_landmarker()

            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                self.timestamps.append(current_time)

                # Convert BGR → RGB and wrap in mp.Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = self._landmarker.detect(mp_image)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]  # list of NormalizedLandmark
                    h, w, _   = frame.shape

                    self._process_eye_tracking(landmarks, w, h, current_time)
                    self._process_head_position(landmarks, w, h)
                    self._process_face_distance(landmarks, w, h)
                    self._process_face_movement(landmarks, w, h, current_time)
                
                self._frames_processed += 1
                time.sleep(0.05)  # ~20 FPS
                
        except Exception as e:
            print(f"Webcam error: {e}. Switching to simulation mode.")
            self._simulation_loop()
        finally:
            if self._cap:
                self._cap.release()
    
    # ── Feature extraction helpers ────────────────────────────────────────

    def _get_point(self, landmarks, idx, w, h):
        """Return pixel (x, y) for a landmark index."""
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _calculate_ear(self, eye_indices, landmarks, w, h):
        """Eye Aspect Ratio using 6 landmark points."""
        pts = [self._get_point(landmarks, i, w, h) for i in eye_indices]
        v1  = self._distance(pts[1], pts[5])
        v2  = self._distance(pts[2], pts[4])
        h1  = self._distance(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h1) if h1 > 0 else 0.0

    def _process_eye_tracking(self, landmarks, w, h, current_time):
        """Process eye tracking for blink detection"""
        left_ear  = self._calculate_ear(self.LEFT_EYE_INDICES,  landmarks, w, h)
        right_ear = self._calculate_ear(self.RIGHT_EYE_INDICES, landmarks, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0
        self.eye_openness.append(avg_ear)

        blink_threshold = 0.2
        if avg_ear < blink_threshold:
            if not self._eye_closed:
                self._eye_closed      = True
                self._eye_close_start = current_time
        else:
            if self._eye_closed:
                if self._eye_close_start:
                    duration = current_time - self._eye_close_start
                    # Valid blink (50ms - 500ms)
                    if 0.05 < duration < 0.5:
                        self.blink_times.append(current_time)
                        self.blink_durations.append(duration)
                        self._blink_count += 1
                self._eye_closed      = False
                self._eye_close_start = None
    
    def _process_head_position(self, landmarks, w, h):
        """Estimate head pose (pitch, yaw, roll)"""
        nose_tip        = landmarks[4]
        left_eye_outer  = landmarks[33]
        right_eye_outer = landmarks[263]

        eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
        eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
        yaw   = (nose_tip.x - eye_center_x) * 100
        pitch = (nose_tip.y - eye_center_y) * 100

        dy   = right_eye_outer.y - left_eye_outer.y
        dx   = right_eye_outer.x - left_eye_outer.x
        roll = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0
        self.head_positions.append({
            'pitch': pitch,
            'yaw':   yaw,
            'roll':  roll,
            'time':  time.time()
        })
    
    def _process_face_distance(self, landmarks, w, h):
        """Estimate face distance from screen"""
        left_face     = landmarks[234]
        right_face    = landmarks[454]
        face_width_px = abs(right_face.x - left_face.x) * w
        if face_width_px > 0:
            # Approximate distance using average face width (~14cm)
            focal_length      = 600
            avg_face_width_cm = 14
            distance_cm = (avg_face_width_cm * focal_length) / face_width_px
            self.face_distances.append(distance_cm)
    
    def _process_face_movement(self, landmarks, w, h, current_time):
        """Track face movement/fidgeting"""
        nose_tip    = landmarks[4]
        current_pos = (nose_tip.x * w, nose_tip.y * h)
        if self._last_face_position:
            movement = self._distance(current_pos, self._last_face_position)
            self.face_movements.append({'movement': movement, 'time': current_time})
        self._last_face_position = current_pos

    # ── Simulation fallback ───────────────────────────────────────────────

    def _simulation_loop(self):
        """Simulation mode for testing without camera"""
        import random
        while self._running:
            current_time = time.time()
            self.timestamps.append(current_time)
            
            # Simulate blinks (normal rate ~15-20 per minute)
            if random.random() < 0.005:  # ~15 blinks per minute at 50ms intervals
                self.blink_times.append(current_time)
                self.blink_durations.append(random.uniform(0.1, 0.3))
                self._blink_count += 1
            
            # Simulate eye openness
            self.eye_openness.append(max(0.1, min(0.5, 0.3 + random.gauss(0, 0.02))))

            # Simulate head position
            self.head_positions.append({
                'pitch': random.gauss(0, 5),
                'yaw':   random.gauss(0, 5),
                'roll':  random.gauss(0, 3),
                'time':  current_time
            })
            
            # Simulate face distance (45-65cm is normal)
            self.face_distances.append(55 + random.gauss(0, 5))
            
            # Simulate face movement
            self.face_movements.append({'movement': abs(random.gauss(0, 2)), 'time': current_time})
            self._frames_processed += 1
            time.sleep(0.05)
    
    # ── Feature extraction ────────────────────────────────────────────────

    def get_features(self):
        """
        Calculate and return current webcam/facial features.
        
        Returns:
            dict: Facial activity features for prediction
        """
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Filter data to current window
        recent_blinks          = [t for t in self.blink_times if t > window_start]
        recent_blink_durations = list(self.blink_durations)[-30:]
        recent_eye_openness    = list(self.eye_openness)[-100:]
        recent_head_positions  = [p for p in self.head_positions if p['time'] > window_start]
        recent_distances       = list(self.face_distances)[-100:]
        recent_movements       = [m for m in self.face_movements if m['time'] > window_start]

        features = {}
        
        # 1. Blink Rate (blinks per minute)
        features['blink_count'] = len(recent_blinks)
        features['blink_rate']  = (len(recent_blinks) / self.window_size) * 60

        # 2. Blink Duration
        features['avg_blink_duration'] = statistics.mean(recent_blink_durations) if recent_blink_durations else 0

        # 3. Eye Openness
        if recent_eye_openness:
            features['avg_eye_openness']      = statistics.mean(recent_eye_openness)
            features['eye_openness_variance'] = statistics.variance(recent_eye_openness) if len(recent_eye_openness) > 1 else 0
        else:
            features['avg_eye_openness'] = features['eye_openness_variance'] = 0

        # 4. Head Position
        if recent_head_positions:
            pitches = [p['pitch'] for p in recent_head_positions]
            yaws    = [p['yaw']   for p in recent_head_positions]
            rolls   = [p['roll']  for p in recent_head_positions]
            features['avg_head_pitch'] = statistics.mean(pitches)
            features['avg_head_yaw']   = statistics.mean(yaws)
            features['avg_head_roll']  = statistics.mean(rolls)
            features['head_movement_variance'] = (
                statistics.variance(pitches) +
                statistics.variance(yaws) +
                statistics.variance(rolls)
            ) / 3 if len(recent_head_positions) > 1 else 0
        else:
            features['avg_head_pitch'] = features['avg_head_yaw'] = features['avg_head_roll'] = 0
            features['head_movement_variance'] = 0
        
        # 5. Face Distance
        if recent_distances:
            features['avg_face_distance']      = statistics.mean(recent_distances)
            features['min_face_distance']      = min(recent_distances)
            features['face_distance_variance'] = statistics.variance(recent_distances) if len(recent_distances) > 1 else 0
        else:
            features['avg_face_distance'] = features['min_face_distance'] = features['face_distance_variance'] = 0

        # 6. Face Movement/Fidgeting
        if recent_movements:
            movements = [m['movement'] for m in recent_movements]
            features['avg_face_movement']   = statistics.mean(movements)
            features['total_face_movement'] = sum(movements)
        else:
            features['avg_face_movement'] = features['total_face_movement'] = 0

        # 7. Deviation from baseline
        features['blink_rate_deviation'] = self._calculate_deviation(
            features['blink_rate'], self.baseline['blink_rate'])
        features['distance_deviation'] = self._calculate_deviation(
            features['avg_face_distance'], self.baseline['face_distance'])

        # 8. Eye strain indicators
        features['eye_strain_score'] = self._calculate_eye_strain(features)
        
        # 9. Overall webcam fatigue score
        features['webcam_fatigue_score'] = self._calculate_fatigue_score(features)
        
        # Metadata
        features['frames_processed'] = self._frames_processed
        features['total_blinks']     = self._blink_count
        features['window_size']      = self.window_size
        features['camera_active']    = OPENCV_AVAILABLE and self._cap is not None

        return features
    
    def _calculate_deviation(self, current, baseline):
        """Calculate percentage deviation from baseline"""
        return (current - baseline) / baseline if baseline != 0 else 0

    def _calculate_eye_strain(self, f):
        """
        Calculate eye strain score based on:
        - Reduced blink rate (dry eyes)
        - Decreased eye openness (squinting)
        - Closer face distance (leaning in)
        - Increased head tilt
        """
        score = 0.0
        # Reduced blink rate (normal is 15-20/min, <10 indicates strain)
        if f['blink_rate'] < 10:
            score += 0.3 * (1 - f['blink_rate'] / 10)
        # Decreased eye openness (squinting)
        if f['avg_eye_openness'] < 0.25:
            score += 0.2 * (1 - f['avg_eye_openness'] / 0.25)
        # Closer face distance (leaning in to see better)
        if 0 < f['avg_face_distance'] < 40:
            score += 0.25 * (1 - f['avg_face_distance'] / 40)
        # Head tilt (neck strain)
        if abs(f['avg_head_roll']) > 10:
            score += 0.25 * min(1, abs(f['avg_head_roll']) / 30)
        return min(1.0, score)

    def _calculate_fatigue_score(self, f):
        """
        Calculate overall webcam-based fatigue score.
        
        Indicators:
        - Reduced blink rate
        - Longer blink duration
        - Decreased eye openness
        - More head movement (fidgeting)
        - Closer to screen
        - Head tilt
        """
        score = 0.0
        # Reduced blink rate
        if f['blink_rate_deviation'] < -0.3:
            score += 0.20 * min(1, abs(f['blink_rate_deviation']))
        # Longer blink duration (drowsiness)
        if f['avg_blink_duration'] > 0.25:
            score += 0.10 * min(1, (f['avg_blink_duration'] - 0.15) / 0.35)
        # Decreased eye openness
        if f['avg_eye_openness'] < 0.25:
            score += 0.20 * (1 - f['avg_eye_openness'] / 0.3)
        # Increased head movement (restlessness)
        if f['head_movement_variance'] > 20:
            score += 0.15 * min(1, f['head_movement_variance'] / 50)
        # Closer to screen
        if f['distance_deviation'] < -0.2:
            score += 0.15 * min(1, abs(f['distance_deviation']))
        # Eye strain component
        score += 0.20 * f['eye_strain_score']
        return min(1.0, score)
    
    def calibrate_baseline(self, duration=60):
        """Calibrate baseline values from normal usage"""
        print(f"Starting webcam baseline calibration for {duration} seconds...")
        print("Please look at the screen normally.")
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(1)
        f = self.get_features()
        self.baseline = {
            'blink_rate':          f['blink_rate']        if f['blink_rate'] > 0        else 15,
            'avg_eye_openness':    f['avg_eye_openness']  if f['avg_eye_openness'] > 0  else 0.3,
            'face_distance':       f['avg_face_distance'] if f['avg_face_distance'] > 0 else 50,
            'head_tilt_threshold': 15,
        }
        print(f"Webcam baseline calibrated: {self.baseline}")
        return self.baseline
    
    def reset_stats(self):
        """Reset all statistics"""
        self.blink_times.clear()
        self.blink_durations.clear()
        self.head_positions.clear()
        self.face_distances.clear()
        self.eye_openness.clear()
        self.face_movements.clear()
        self.timestamps.clear()
        self._frames_processed = 0
        self._blink_count = 0
