"""
Keyboard Activity Monitor
Tracks typing patterns to detect cognitive slowdown and fatigue indicators.

Features tracked:
- Typing speed (keys per minute)
- Pauses between keystrokes
- Error frequency (backspace usage)
- Key hold duration
- Typing rhythm consistency
"""

import threading
import time
from datetime import datetime, timedelta
from collections import deque
import statistics

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Keyboard monitoring will use simulation mode.")


class KeyboardMonitor:
    """Monitors keyboard activity for migraine prediction features"""
    
    def __init__(self, window_size=60):
        """
        Initialize keyboard monitor.
        
        Args:
            window_size: Time window in seconds for calculating features
        """
        self.window_size = window_size
        self._running = False
        self._thread = None
        self._listener = None
        
        # Data storage using deques for efficient sliding window
        self.keypress_times = deque(maxlen=1000)  # Timestamps of keypresses
        self.key_intervals = deque(maxlen=500)     # Time between keypresses
        self.key_hold_durations = deque(maxlen=500)  # How long keys are held
        self.backspace_times = deque(maxlen=200)   # Timestamps of backspaces (errors)
        self.pause_durations = deque(maxlen=100)   # Long pauses (>2 seconds)
        
        # State tracking
        self._last_keypress_time = None
        self._key_press_start = {}  # Track when each key was pressed
        self._total_keystrokes = 0
        self._total_errors = 0
        
        # Baseline values (will be calibrated)
        self.baseline = {
            'typing_speed': 200,  # keys per minute
            'avg_interval': 0.15,  # seconds between keys
            'error_rate': 0.05,   # 5% error rate
            'avg_hold_duration': 0.1  # seconds
        }
    
    def start(self):
        """Start keyboard monitoring"""
        if self._running:
            return
        
        self._running = True
        
        if PYNPUT_AVAILABLE:
            self._listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self._listener.start()
        else:
            # Simulation mode for testing
            self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self._thread.start()
        
        print("[OK] Keyboard monitor started")
    
    def stop(self):
        """Stop keyboard monitoring"""
        self._running = False
        
        if self._listener:
            self._listener.stop()
            self._listener = None
        
        print("[OK] Keyboard monitor stopped")
    
    def is_running(self):
        """Check if monitor is running"""
        return self._running
    
    def _on_key_press(self, key):
        """Handle key press event"""
        if not self._running:
            return
        
        current_time = time.time()
        self._total_keystrokes += 1
        
        # Record keypress time
        self.keypress_times.append(current_time)
        
        # Calculate interval from last keypress
        if self._last_keypress_time:
            interval = current_time - self._last_keypress_time
            
            # Check for pause (>2 seconds indicates distraction/fatigue)
            if interval > 2.0:
                self.pause_durations.append(interval)
            else:
                self.key_intervals.append(interval)
        
        self._last_keypress_time = current_time
        
        # Track key hold start time
        key_id = str(key)
        self._key_press_start[key_id] = current_time
        
        # Check for backspace (error indicator)
        try:
            if key == keyboard.Key.backspace:
                self.backspace_times.append(current_time)
                self._total_errors += 1
        except:
            pass
    
    def _on_key_release(self, key):
        """Handle key release event"""
        if not self._running:
            return
        
        current_time = time.time()
        key_id = str(key)
        
        # Calculate key hold duration
        if key_id in self._key_press_start:
            hold_duration = current_time - self._key_press_start[key_id]
            self.key_hold_durations.append(hold_duration)
            del self._key_press_start[key_id]
    
    def _simulation_loop(self):
        """Simulation mode for testing without actual keyboard input"""
        import random
        
        while self._running:
            # Simulate typing activity
            current_time = time.time()
            
            # Simulate keypress
            self.keypress_times.append(current_time)
            self._total_keystrokes += 1
            
            # Simulate interval (with some variation to simulate fatigue)
            base_interval = 0.15 + random.gauss(0, 0.05)
            self.key_intervals.append(max(0.05, base_interval))
            
            # Simulate hold duration
            hold = 0.1 + random.gauss(0, 0.02)
            self.key_hold_durations.append(max(0.05, hold))
            
            # Occasional backspace (error)
            if random.random() < 0.05:
                self.backspace_times.append(current_time)
                self._total_errors += 1
            
            # Occasional pause
            if random.random() < 0.02:
                self.pause_durations.append(random.uniform(2, 5))
            
            self._last_keypress_time = current_time
            
            # Wait before next simulated keypress
            time.sleep(random.uniform(0.1, 0.3))
    
    def get_features(self):
        """
        Calculate and return current keyboard features.
        
        Returns:
            dict: Keyboard activity features for prediction
        """
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Filter data to current window
        recent_keypresses = [t for t in self.keypress_times if t > window_start]
        recent_intervals = list(self.key_intervals)[-50:]  # Last 50 intervals
        recent_holds = list(self.key_hold_durations)[-50:]
        recent_errors = [t for t in self.backspace_times if t > window_start]
        recent_pauses = list(self.pause_durations)[-20:]
        
        # Calculate features
        features = {}
        
        # 1. Typing Speed (keys per minute)
        if len(recent_keypresses) >= 2:
            time_span = recent_keypresses[-1] - recent_keypresses[0]
            if time_span > 0:
                features['typing_speed'] = (len(recent_keypresses) / time_span) * 60
            else:
                features['typing_speed'] = 0
        else:
            features['typing_speed'] = 0
        
        # 2. Average interval between keystrokes
        if recent_intervals:
            features['avg_interval'] = statistics.mean(recent_intervals)
            features['interval_variance'] = statistics.variance(recent_intervals) if len(recent_intervals) > 1 else 0
        else:
            features['avg_interval'] = 0
            features['interval_variance'] = 0
        
        # 3. Error rate (backspaces per keystroke)
        total_recent = len(recent_keypresses)
        if total_recent > 0:
            features['error_rate'] = len(recent_errors) / total_recent
        else:
            features['error_rate'] = 0
        
        # 4. Key hold duration
        if recent_holds:
            features['avg_hold_duration'] = statistics.mean(recent_holds)
            features['hold_variance'] = statistics.variance(recent_holds) if len(recent_holds) > 1 else 0
        else:
            features['avg_hold_duration'] = 0
            features['hold_variance'] = 0
        
        # 5. Pause frequency and duration
        features['pause_count'] = len(recent_pauses)
        features['avg_pause_duration'] = statistics.mean(recent_pauses) if recent_pauses else 0
        
        # 6. Typing rhythm consistency (coefficient of variation)
        if recent_intervals and len(recent_intervals) > 1:
            mean_interval = statistics.mean(recent_intervals)
            if mean_interval > 0:
                features['rhythm_consistency'] = statistics.stdev(recent_intervals) / mean_interval
            else:
                features['rhythm_consistency'] = 0
        else:
            features['rhythm_consistency'] = 0
        
        # 7. Activity level (keypresses in window)
        features['activity_level'] = len(recent_keypresses)
        
        # 8. Fatigue indicators (compared to baseline)
        features['speed_deviation'] = self._calculate_deviation(
            features['typing_speed'], 
            self.baseline['typing_speed']
        )
        features['interval_deviation'] = self._calculate_deviation(
            features['avg_interval'],
            self.baseline['avg_interval']
        )
        features['error_deviation'] = self._calculate_deviation(
            features['error_rate'],
            self.baseline['error_rate']
        )
        
        # 9. Overall keyboard fatigue score (0-1)
        features['keyboard_fatigue_score'] = self._calculate_fatigue_score(features)
        
        # Metadata
        features['total_keystrokes'] = self._total_keystrokes
        features['total_errors'] = self._total_errors
        features['window_size'] = self.window_size
        
        return features
    
    def _calculate_deviation(self, current, baseline):
        """Calculate percentage deviation from baseline"""
        if baseline == 0:
            return 0
        return (current - baseline) / baseline
    
    def _calculate_fatigue_score(self, features):
        """
        Calculate overall keyboard fatigue score.
        
        Indicators of fatigue:
        - Slower typing speed
        - Longer intervals between keys
        - Higher error rate
        - Longer key hold durations
        - More pauses
        - Less consistent rhythm
        """
        score = 0.0
        weights = {
            'speed': 0.2,
            'interval': 0.15,
            'error': 0.25,
            'hold': 0.1,
            'pause': 0.15,
            'rhythm': 0.15
        }
        
        # Speed decrease (negative deviation = slower = more fatigue)
        if features['speed_deviation'] < -0.2:
            score += weights['speed'] * min(1, abs(features['speed_deviation']))
        
        # Interval increase (positive deviation = slower = more fatigue)
        if features['interval_deviation'] > 0.2:
            score += weights['interval'] * min(1, features['interval_deviation'])
        
        # Error rate increase
        if features['error_deviation'] > 0.3:
            score += weights['error'] * min(1, features['error_deviation'])
        
        # Hold duration increase
        if features['avg_hold_duration'] > self.baseline['avg_hold_duration'] * 1.3:
            hold_increase = (features['avg_hold_duration'] - self.baseline['avg_hold_duration']) / self.baseline['avg_hold_duration']
            score += weights['hold'] * min(1, hold_increase)
        
        # Pause frequency
        if features['pause_count'] > 3:
            score += weights['pause'] * min(1, features['pause_count'] / 10)
        
        # Rhythm inconsistency
        if features['rhythm_consistency'] > 0.5:
            score += weights['rhythm'] * min(1, features['rhythm_consistency'])
        
        return min(1.0, score)
    
    def calibrate_baseline(self, duration=300):
        """
        Calibrate baseline values from user's normal typing.
        Should be run when user is in good condition.
        
        Args:
            duration: Calibration duration in seconds (default 5 minutes)
        """
        print(f"Starting baseline calibration for {duration} seconds...")
        print("Please type normally during this period.")
        
        # Clear existing data
        self.keypress_times.clear()
        self.key_intervals.clear()
        self.key_hold_durations.clear()
        self.backspace_times.clear()
        
        # Collect data for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(1)
        
        # Calculate baseline values
        features = self.get_features()
        
        self.baseline = {
            'typing_speed': features['typing_speed'] if features['typing_speed'] > 0 else 200,
            'avg_interval': features['avg_interval'] if features['avg_interval'] > 0 else 0.15,
            'error_rate': features['error_rate'] if features['error_rate'] > 0 else 0.05,
            'avg_hold_duration': features['avg_hold_duration'] if features['avg_hold_duration'] > 0 else 0.1
        }
        
        print(f"Baseline calibrated: {self.baseline}")
        return self.baseline
    
    def reset_stats(self):
        """Reset all statistics"""
        self.keypress_times.clear()
        self.key_intervals.clear()
        self.key_hold_durations.clear()
        self.backspace_times.clear()
        self.pause_durations.clear()
        self._total_keystrokes = 0
        self._total_errors = 0
        self._last_keypress_time = None
