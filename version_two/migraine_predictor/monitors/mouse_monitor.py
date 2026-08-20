"""
Mouse Activity Monitor
Tracks mouse movement patterns to detect attention and fatigue indicators.

Features tracked:
- Movement speed and patterns
- Click frequency and types
- Scroll activity
- Periods of inactivity
- Movement smoothness/jitter
"""

import threading
import time
import math
from datetime import datetime
from collections import deque
import statistics

try:
    from pynput import mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Mouse monitoring will use simulation mode.")


class MouseMonitor:
    """Monitors mouse activity for migraine prediction features"""
    
    def __init__(self, window_size=60):
        """
        Initialize mouse monitor.
        
        Args:
            window_size: Time window in seconds for calculating features
        """
        self.window_size = window_size
        self._running = False
        self._thread = None
        self._listener = None
        
        # Position tracking
        self._last_position = None
        self._last_move_time = None
        
        # Data storage using deques
        self.movement_speeds = deque(maxlen=500)      # Pixels per second
        self.movement_distances = deque(maxlen=500)   # Distance of each movement
        self.movement_times = deque(maxlen=500)       # Timestamps of movements
        self.click_times = deque(maxlen=200)          # Timestamps of clicks
        self.click_types = deque(maxlen=200)          # left, right, middle
        self.scroll_events = deque(maxlen=200)        # Scroll timestamps and amounts
        self.idle_periods = deque(maxlen=100)         # Duration of idle periods
        self.direction_changes = deque(maxlen=300)    # Sudden direction changes (jitter)
        
        # State tracking
        self._total_distance = 0.0
        self._total_clicks = 0
        self._total_scrolls = 0
        self._last_direction = None
        self._idle_start = None
        
        # Baseline values
        self.baseline = {
            'avg_speed': 500,          # pixels per second
            'click_rate': 10,          # clicks per minute
            'scroll_rate': 5,          # scrolls per minute
            'idle_threshold': 3,       # seconds before considered idle
            'jitter_threshold': 0.3    # direction change threshold
        }
    
    def start(self):
        """Start mouse monitoring"""
        if self._running:
            return
        
        self._running = True
        
        if PYNPUT_AVAILABLE:
            self._listener = mouse.Listener(
                on_move=self._on_move,
                on_click=self._on_click,
                on_scroll=self._on_scroll
            )
            self._listener.start()
        else:
            # Simulation mode
            self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self._thread.start()
        
        print("[OK] Mouse monitor started")
    
    def stop(self):
        """Stop mouse monitoring"""
        self._running = False
        
        if self._listener:
            self._listener.stop()
            self._listener = None
        
        print("[OK] Mouse monitor stopped")
    
    def is_running(self):
        """Check if monitor is running"""
        return self._running
    
    def _on_move(self, x, y):
        """Handle mouse movement event"""
        if not self._running:
            return
        
        current_time = time.time()
        current_position = (x, y)
        
        # Check for idle period ending
        if self._idle_start:
            idle_duration = current_time - self._idle_start
            if idle_duration > self.baseline['idle_threshold']:
                self.idle_periods.append(idle_duration)
            self._idle_start = None
        
        if self._last_position and self._last_move_time:
            # Calculate distance
            dx = x - self._last_position[0]
            dy = y - self._last_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate time delta
            time_delta = current_time - self._last_move_time
            
            if time_delta > 0 and distance > 0:
                # Calculate speed
                speed = distance / time_delta
                self.movement_speeds.append(speed)
                self.movement_distances.append(distance)
                self.movement_times.append(current_time)
                
                self._total_distance += distance
                
                # Calculate direction and check for jitter
                if distance > 5:  # Ignore tiny movements
                    current_direction = math.atan2(dy, dx)
                    
                    if self._last_direction is not None:
                        direction_change = abs(current_direction - self._last_direction)
                        # Normalize to [0, pi]
                        if direction_change > math.pi:
                            direction_change = 2 * math.pi - direction_change
                        
                        # Record significant direction changes (potential jitter)
                        if direction_change > self.baseline['jitter_threshold']:
                            self.direction_changes.append({
                                'time': current_time,
                                'change': direction_change
                            })
                    
                    self._last_direction = current_direction
        
        self._last_position = current_position
        self._last_move_time = current_time
    
    def _on_click(self, x, y, button, pressed):
        """Handle mouse click event"""
        if not self._running or not pressed:
            return
        
        current_time = time.time()
        self.click_times.append(current_time)
        self.click_types.append(str(button))
        self._total_clicks += 1
    
    def _on_scroll(self, x, y, dx, dy):
        """Handle scroll event"""
        if not self._running:
            return
        
        current_time = time.time()
        self.scroll_events.append({
            'time': current_time,
            'dx': dx,
            'dy': dy
        })
        self._total_scrolls += 1
    
    def _simulation_loop(self):
        """Simulation mode for testing"""
        import random
        
        x, y = 500, 500
        
        while self._running:
            current_time = time.time()
            
            # Simulate movement
            dx = random.gauss(0, 50)
            dy = random.gauss(0, 50)
            x = max(0, min(1920, x + dx))
            y = max(0, min(1080, y + dy))
            
            distance = math.sqrt(dx*dx + dy*dy)
            speed = distance / 0.05  # Assuming 50ms between updates
            
            self.movement_speeds.append(speed)
            self.movement_distances.append(distance)
            self.movement_times.append(current_time)
            self._total_distance += distance
            
            # Occasional click
            if random.random() < 0.05:
                self.click_times.append(current_time)
                self.click_types.append('left')
                self._total_clicks += 1
            
            # Occasional scroll
            if random.random() < 0.03:
                self.scroll_events.append({
                    'time': current_time,
                    'dx': 0,
                    'dy': random.choice([-1, 1])
                })
                self._total_scrolls += 1
            
            # Occasional idle period
            if random.random() < 0.01:
                self.idle_periods.append(random.uniform(3, 10))
            
            # Occasional jitter
            if random.random() < 0.02:
                self.direction_changes.append({
                    'time': current_time,
                    'change': random.uniform(0.5, math.pi)
                })
            
            time.sleep(0.05)
    
    def _check_idle(self):
        """Check and record idle periods"""
        current_time = time.time()
        
        if self._last_move_time:
            idle_time = current_time - self._last_move_time
            
            if idle_time > self.baseline['idle_threshold']:
                if not self._idle_start:
                    self._idle_start = self._last_move_time
    
    def get_features(self):
        """
        Calculate and return current mouse features.
        
        Returns:
            dict: Mouse activity features for prediction
        """
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Check for current idle
        self._check_idle()
        
        # Filter data to current window
        recent_speeds = [s for s, t in zip(self.movement_speeds, self.movement_times) 
                        if t > window_start] if self.movement_times else []
        recent_distances = list(self.movement_distances)[-50:]
        recent_clicks = [t for t in self.click_times if t > window_start]
        recent_scrolls = [s for s in self.scroll_events if s['time'] > window_start]
        recent_idles = list(self.idle_periods)[-20:]
        recent_jitter = [j for j in self.direction_changes if j['time'] > window_start]
        
        features = {}
        
        # 1. Movement Speed
        if recent_speeds:
            features['avg_speed'] = statistics.mean(recent_speeds)
            features['max_speed'] = max(recent_speeds)
            features['speed_variance'] = statistics.variance(recent_speeds) if len(recent_speeds) > 1 else 0
        else:
            features['avg_speed'] = 0
            features['max_speed'] = 0
            features['speed_variance'] = 0
        
        # 2. Movement Distance
        if recent_distances:
            features['avg_distance'] = statistics.mean(recent_distances)
            features['total_distance_window'] = sum(recent_distances)
        else:
            features['avg_distance'] = 0
            features['total_distance_window'] = 0
        
        # 3. Click Rate (clicks per minute)
        features['click_count'] = len(recent_clicks)
        features['click_rate'] = (len(recent_clicks) / self.window_size) * 60
        
        # 4. Scroll Activity
        features['scroll_count'] = len(recent_scrolls)
        features['scroll_rate'] = (len(recent_scrolls) / self.window_size) * 60
        
        # 5. Idle Periods
        features['idle_count'] = len(recent_idles)
        features['avg_idle_duration'] = statistics.mean(recent_idles) if recent_idles else 0
        features['total_idle_time'] = sum(recent_idles) if recent_idles else 0
        
        # Current idle status
        if self._last_move_time:
            current_idle = current_time - self._last_move_time
            features['current_idle_duration'] = current_idle if current_idle > self.baseline['idle_threshold'] else 0
        else:
            features['current_idle_duration'] = 0
        
        # 6. Movement Smoothness (jitter detection)
        features['jitter_count'] = len(recent_jitter)
        if recent_jitter:
            features['avg_jitter'] = statistics.mean([j['change'] for j in recent_jitter])
        else:
            features['avg_jitter'] = 0
        
        # 7. Movement patterns
        features['movement_activity'] = len([t for t in self.movement_times if t > window_start])
        
        # 8. Deviation from baseline
        features['speed_deviation'] = self._calculate_deviation(
            features['avg_speed'],
            self.baseline['avg_speed']
        )
        features['click_deviation'] = self._calculate_deviation(
            features['click_rate'],
            self.baseline['click_rate']
        )
        
        # 9. Overall mouse fatigue score
        features['mouse_fatigue_score'] = self._calculate_fatigue_score(features)
        
        # Metadata
        features['total_distance'] = self._total_distance
        features['total_clicks'] = self._total_clicks
        features['total_scrolls'] = self._total_scrolls
        features['window_size'] = self.window_size
        
        return features
    
    def _calculate_deviation(self, current, baseline):
        """Calculate percentage deviation from baseline"""
        if baseline == 0:
            return 0
        return (current - baseline) / baseline
    
    def _calculate_fatigue_score(self, features):
        """
        Calculate overall mouse fatigue score.
        
        Indicators of fatigue:
        - Slower movement speed
        - Less clicking activity
        - More idle periods
        - More jitter (shaky movements)
        - Reduced scrolling
        """
        score = 0.0
        weights = {
            'speed': 0.2,
            'click': 0.15,
            'idle': 0.25,
            'jitter': 0.25,
            'activity': 0.15
        }
        
        # Speed decrease
        if features['speed_deviation'] < -0.3:
            score += weights['speed'] * min(1, abs(features['speed_deviation']))
        
        # Click rate decrease
        if features['click_deviation'] < -0.3:
            score += weights['click'] * min(1, abs(features['click_deviation']))
        
        # Idle time increase
        if features['total_idle_time'] > 10:
            idle_factor = features['total_idle_time'] / self.window_size
            score += weights['idle'] * min(1, idle_factor)
        
        # Jitter increase (shaky movements indicate fatigue/tension)
        if features['jitter_count'] > 5:
            score += weights['jitter'] * min(1, features['jitter_count'] / 20)
        
        # Low activity
        if features['movement_activity'] < 10:
            score += weights['activity'] * (1 - features['movement_activity'] / 50)
        
        return min(1.0, score)
    
    def calibrate_baseline(self, duration=300):
        """Calibrate baseline values from normal usage"""
        print(f"Starting mouse baseline calibration for {duration} seconds...")
        
        # Clear existing data
        self.movement_speeds.clear()
        self.click_times.clear()
        self.scroll_events.clear()
        
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(1)
        
        features = self.get_features()
        
        self.baseline = {
            'avg_speed': features['avg_speed'] if features['avg_speed'] > 0 else 500,
            'click_rate': features['click_rate'] if features['click_rate'] > 0 else 10,
            'scroll_rate': features['scroll_rate'] if features['scroll_rate'] > 0 else 5,
            'idle_threshold': 3,
            'jitter_threshold': 0.3
        }
        
        print(f"Mouse baseline calibrated: {self.baseline}")
        return self.baseline
    
    def reset_stats(self):
        """Reset all statistics"""
        self.movement_speeds.clear()
        self.movement_distances.clear()
        self.movement_times.clear()
        self.click_times.clear()
        self.click_types.clear()
        self.scroll_events.clear()
        self.idle_periods.clear()
        self.direction_changes.clear()
        self._total_distance = 0.0
        self._total_clicks = 0
        self._total_scrolls = 0
