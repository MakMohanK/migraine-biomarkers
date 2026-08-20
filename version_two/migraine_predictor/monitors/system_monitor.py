"""
System Usage Monitor
Tracks overall system usage patterns to detect stress and fatigue indicators.

Features tracked:
- Screen time duration
- App switching frequency
- CPU and memory usage
- Idle time
- Late night usage
- Brightness levels (when available)
"""

import threading
import time
from datetime import datetime, timedelta
from collections import deque
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System monitoring will use simulation mode.")


class SystemMonitor:
    """Monitors system usage patterns for migraine prediction"""
    
    def __init__(self, window_size=60, sample_interval=5):
        """
        Initialize system monitor.
        
        Args:
            window_size: Time window in seconds for calculating features
            sample_interval: How often to sample system stats (seconds)
        """
        self.window_size = window_size
        self.sample_interval = sample_interval
        self._running = False
        self._thread = None
        
        # Data storage
        self.cpu_readings = deque(maxlen=500)          # CPU usage percentages
        self.memory_readings = deque(maxlen=500)       # Memory usage percentages
        self.active_windows = deque(maxlen=200)        # Active window titles/apps
        self.app_switches = deque(maxlen=200)          # Timestamps of app switches
        self.idle_periods = deque(maxlen=100)          # System idle periods
        self.timestamps = deque(maxlen=500)            # Reading timestamps
        
        # Session tracking
        self._session_start = None
        self._last_active_window = None
        self._last_activity_time = None
        self._total_app_switches = 0
        
        # Baseline values
        self.baseline = {
            'avg_cpu': 30,              # Average CPU usage %
            'avg_memory': 60,           # Average memory usage %
            'app_switch_rate': 2,       # Switches per minute
            'idle_threshold': 60,       # Seconds before considered idle
            'late_night_start': 23,     # 11 PM
            'late_night_end': 6         # 6 AM
        }
    
    def start(self):
        """Start system monitoring"""
        if self._running:
            return
        
        self._running = True
        self._session_start = datetime.now()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        print("[OK] System monitor started")
    
    def stop(self):
        """Stop system monitoring"""
        self._running = False
        print("[OK] System monitor stopped")
    
    def is_running(self):
        """Check if monitor is running"""
        return self._running
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            current_time = time.time()
            self.timestamps.append(current_time)
            
            if PSUTIL_AVAILABLE:
                self._collect_system_stats()
                self._track_active_window()
            else:
                self._simulation_collect()
            
            time.sleep(self.sample_interval)
    
    def _collect_system_stats(self):
        """Collect CPU, memory, and other system statistics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_readings.append({
            'value': cpu_percent,
            'time': time.time()
        })
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_readings.append({
            'value': memory.percent,
            'time': time.time()
        })
    
    def _track_active_window(self):
        """Track active window/application changes"""
        current_time = time.time()
        
        try:
            # Try to get active window (platform-specific)
            active_window = self._get_active_window()
            
            if active_window and active_window != self._last_active_window:
                # App switch detected
                self.app_switches.append(current_time)
                self._total_app_switches += 1
                
                self.active_windows.append({
                    'window': active_window,
                    'time': current_time
                })
                
                self._last_active_window = active_window
            
            self._last_activity_time = current_time
            
        except Exception as e:
            # Fallback if window tracking fails
            pass
    
    def _get_active_window(self):
        """Get the currently active window title (platform-specific)"""
        import platform
        system = platform.system()
        
        try:
            if system == 'Windows':
                return self._get_active_window_windows()
            elif system == 'Darwin':  # macOS
                return self._get_active_window_macos()
            elif system == 'Linux':
                return self._get_active_window_linux()
        except:
            pass
        
        return None
    
    def _get_active_window_windows(self):
        """Get active window on Windows"""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            h_wnd = user32.GetForegroundWindow()
            
            length = user32.GetWindowTextLengthW(h_wnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(h_wnd, buf, length + 1)
            
            return buf.value if buf.value else None
        except:
            return None
    
    def _get_active_window_macos(self):
        """Get active window on macOS"""
        try:
            from AppKit import NSWorkspace
            active_app = NSWorkspace.sharedWorkspace().activeApplication()
            return active_app.get('NSApplicationName', None)
        except:
            return None
    
    def _get_active_window_linux(self):
        """Get active window on Linux"""
        try:
            import subprocess
            result = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowname'],
                capture_output=True, text=True, timeout=1
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _simulation_collect(self):
        """Simulation mode for testing"""
        import random
        
        current_time = time.time()
        
        # Simulate CPU usage (20-80%)
        cpu = 30 + random.gauss(0, 15)
        self.cpu_readings.append({
            'value': max(5, min(100, cpu)),
            'time': current_time
        })
        
        # Simulate memory usage (40-80%)
        memory = 60 + random.gauss(0, 10)
        self.memory_readings.append({
            'value': max(20, min(95, memory)),
            'time': current_time
        })
        
        # Simulate occasional app switches
        if random.random() < 0.1:
            self.app_switches.append(current_time)
            self._total_app_switches += 1
            
            apps = ['Chrome', 'VSCode', 'Slack', 'Terminal', 'Finder', 'Notes']
            self.active_windows.append({
                'window': random.choice(apps),
                'time': current_time
            })
        
        # Simulate occasional idle periods
        if random.random() < 0.02:
            self.idle_periods.append({
                'duration': random.uniform(60, 300),
                'time': current_time
            })
    
    def get_features(self):
        """
        Calculate and return current system usage features.
        
        Returns:
            dict: System usage features for prediction
        """
        current_time = time.time()
        window_start = current_time - self.window_size
        now = datetime.now()
        
        # Filter data to current window
        recent_cpu = [r for r in self.cpu_readings if r['time'] > window_start]
        recent_memory = [r for r in self.memory_readings if r['time'] > window_start]
        recent_switches = [t for t in self.app_switches if t > window_start]
        recent_windows = [w for w in self.active_windows if w['time'] > window_start]
        
        features = {}
        
        # 1. CPU Usage
        if recent_cpu:
            cpu_values = [r['value'] for r in recent_cpu]
            features['avg_cpu'] = statistics.mean(cpu_values)
            features['max_cpu'] = max(cpu_values)
            features['cpu_variance'] = statistics.variance(cpu_values) if len(cpu_values) > 1 else 0
        else:
            features['avg_cpu'] = 0
            features['max_cpu'] = 0
            features['cpu_variance'] = 0
        
        # 2. Memory Usage
        if recent_memory:
            memory_values = [r['value'] for r in recent_memory]
            features['avg_memory'] = statistics.mean(memory_values)
            features['max_memory'] = max(memory_values)
            features['memory_variance'] = statistics.variance(memory_values) if len(memory_values) > 1 else 0
        else:
            features['avg_memory'] = 0
            features['max_memory'] = 0
            features['memory_variance'] = 0
        
        # 3. App Switching
        features['app_switch_count'] = len(recent_switches)
        features['app_switch_rate'] = (len(recent_switches) / self.window_size) * 60  # per minute
        
        # 4. Unique apps used
        if recent_windows:
            unique_apps = set(w['window'] for w in recent_windows if w['window'])
            features['unique_apps'] = len(unique_apps)
        else:
            features['unique_apps'] = 0
        
        # 5. Session Duration
        if self._session_start:
            session_duration = (now - self._session_start).total_seconds()
            features['session_duration_minutes'] = session_duration / 60
        else:
            features['session_duration_minutes'] = 0
        
        # 6. Time of Day Features
        features['hour_of_day'] = now.hour
        features['is_late_night'] = self._is_late_night(now.hour)
        features['is_work_hours'] = 9 <= now.hour <= 18
        
        # 7. System Load Indicators
        features['high_cpu_periods'] = sum(1 for r in recent_cpu if r['value'] > 80)
        features['high_memory_periods'] = sum(1 for r in recent_memory if r['value'] > 85)
        
        # 8. Deviation from baseline
        features['cpu_deviation'] = self._calculate_deviation(
            features['avg_cpu'],
            self.baseline['avg_cpu']
        )
        features['memory_deviation'] = self._calculate_deviation(
            features['avg_memory'],
            self.baseline['avg_memory']
        )
        features['switch_rate_deviation'] = self._calculate_deviation(
            features['app_switch_rate'],
            self.baseline['app_switch_rate']
        )
        
        # 9. Workload Score
        features['workload_score'] = self._calculate_workload_score(features)
        
        # 10. Overall system fatigue score
        features['system_fatigue_score'] = self._calculate_fatigue_score(features)
        
        # Metadata
        features['total_app_switches'] = self._total_app_switches
        features['window_size'] = self.window_size
        features['monitoring_available'] = PSUTIL_AVAILABLE
        
        return features
    
    def _calculate_deviation(self, current, baseline):
        """Calculate percentage deviation from baseline"""
        if baseline == 0:
            return 0
        return (current - baseline) / baseline
    
    def _is_late_night(self, hour):
        """Check if current hour is during late night"""
        start = self.baseline['late_night_start']
        end = self.baseline['late_night_end']
        
        if start > end:  # Crosses midnight
            return hour >= start or hour < end
        return start <= hour < end
    
    def _calculate_workload_score(self, features):
        """
        Calculate workload intensity score based on:
        - CPU usage
        - Memory usage
        - App switching frequency
        - Number of apps used
        """
        score = 0.0
        
        # High CPU indicates heavy workload
        if features['avg_cpu'] > 50:
            score += 0.25 * min(1, (features['avg_cpu'] - 50) / 50)
        
        # High memory indicates many tasks
        if features['avg_memory'] > 70:
            score += 0.25 * min(1, (features['avg_memory'] - 70) / 30)
        
        # Frequent app switching indicates multitasking
        if features['app_switch_rate'] > 3:
            score += 0.25 * min(1, features['app_switch_rate'] / 10)
        
        # Many apps open indicates complex work
        if features['unique_apps'] > 5:
            score += 0.25 * min(1, features['unique_apps'] / 15)
        
        return min(1.0, score)
    
    def _calculate_fatigue_score(self, features):
        """
        Calculate overall system-based fatigue score.
        
        Indicators:
        - Long session duration
        - Late night usage
        - High system load (stress indicator)
        - Frequent app switching (distraction)
        - High CPU/memory sustained (demanding work)
        """
        score = 0.0
        weights = {
            'session': 0.2,
            'late_night': 0.25,
            'workload': 0.2,
            'switching': 0.15,
            'system_load': 0.2
        }
        
        # Long session without breaks
        if features['session_duration_minutes'] > 60:
            hours = features['session_duration_minutes'] / 60
            score += weights['session'] * min(1, hours / 4)  # Max at 4 hours
        
        # Late night usage
        if features['is_late_night']:
            score += weights['late_night'] * 0.8
        
        # High workload
        score += weights['workload'] * features['workload_score']
        
        # Excessive app switching (distraction/stress)
        if features['switch_rate_deviation'] > 0.5:
            score += weights['switching'] * min(1, features['switch_rate_deviation'])
        
        # Sustained high system load
        high_load_ratio = (features['high_cpu_periods'] + features['high_memory_periods']) / max(1, len(self.timestamps))
        score += weights['system_load'] * min(1, high_load_ratio * 2)
        
        return min(1.0, score)
    
    def get_session_stats(self):
        """Get statistics for the current session"""
        if not self._session_start:
            return {}
        
        now = datetime.now()
        duration = now - self._session_start
        
        return {
            'session_start': self._session_start.isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'total_app_switches': self._total_app_switches,
            'avg_switches_per_hour': (self._total_app_switches / max(1, duration.total_seconds())) * 3600
        }
    
    def calibrate_baseline(self, duration=300):
        """Calibrate baseline values from normal usage"""
        print(f"Starting system baseline calibration for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(1)
        
        features = self.get_features()
        
        self.baseline = {
            'avg_cpu': features['avg_cpu'] if features['avg_cpu'] > 0 else 30,
            'avg_memory': features['avg_memory'] if features['avg_memory'] > 0 else 60,
            'app_switch_rate': features['app_switch_rate'] if features['app_switch_rate'] > 0 else 2,
            'idle_threshold': 60,
            'late_night_start': 23,
            'late_night_end': 6
        }
        
        print(f"System baseline calibrated: {self.baseline}")
        return self.baseline
    
    def reset_stats(self):
        """Reset all statistics"""
        self.cpu_readings.clear()
        self.memory_readings.clear()
        self.active_windows.clear()
        self.app_switches.clear()
        self.idle_periods.clear()
        self.timestamps.clear()
        self._total_app_switches = 0
        self._session_start = datetime.now()
