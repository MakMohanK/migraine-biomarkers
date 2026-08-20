"""
Database Models for Migraine Predictor System
Stores activity logs, predictions, and user sessions.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()


class ActivityLog(db.Model):
    """Stores collected activity data from all monitors"""
    __tablename__ = 'activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Feature data stored as JSON strings
    keyboard_features = db.Column(db.Text, default='{}')
    mouse_features = db.Column(db.Text, default='{}')
    system_features = db.Column(db.Text, default='{}')
    webcam_features = db.Column(db.Text, default='{}')
    
    # Calculated risk score at time of logging
    risk_score = db.Column(db.Float, default=0.0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'keyboard_features': json.loads(self.keyboard_features) if self.keyboard_features else {},
            'mouse_features': json.loads(self.mouse_features) if self.mouse_features else {},
            'system_features': json.loads(self.system_features) if self.system_features else {},
            'webcam_features': json.loads(self.webcam_features) if self.webcam_features else {},
            'risk_score': self.risk_score
        }


class MigrainePrediction(db.Model):
    """Stores migraine predictions and confirmed migraines"""
    __tablename__ = 'migraine_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Prediction results
    risk_score = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    risk_level = db.Column(db.String(20), default='low')  # low, moderate, high, critical
    
    # Migraine classification (Phase 2)
    migraine_type = db.Column(db.String(50), nullable=True)  # tension, cluster, migraine_with_aura, etc.
    intensity = db.Column(db.Integer, nullable=True)  # 1-10 scale
    
    # User feedback
    is_confirmed = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)
    
    # Contributing factors (JSON)
    contributing_factors = db.Column(db.Text, default='{}')
    
    # Therapy recommendations given
    therapy_recommendations = db.Column(db.Text, default='{}')
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'migraine_type': self.migraine_type,
            'intensity': self.intensity,
            'is_confirmed': self.is_confirmed,
            'notes': self.notes,
            'contributing_factors': json.loads(self.contributing_factors) if self.contributing_factors else {},
            'therapy_recommendations': json.loads(self.therapy_recommendations) if self.therapy_recommendations else {}
        }


class UserSession(db.Model):
    """Tracks user sessions for analysis"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    
    # Session statistics
    total_keystrokes = db.Column(db.Integer, default=0)
    total_mouse_clicks = db.Column(db.Integer, default=0)
    total_mouse_distance = db.Column(db.Float, default=0.0)
    avg_typing_speed = db.Column(db.Float, default=0.0)
    
    # Risk tracking during session
    max_risk_score = db.Column(db.Float, default=0.0)
    avg_risk_score = db.Column(db.Float, default=0.0)
    
    # Did migraine occur during/after session?
    migraine_occurred = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_keystrokes': self.total_keystrokes,
            'total_mouse_clicks': self.total_mouse_clicks,
            'total_mouse_distance': self.total_mouse_distance,
            'avg_typing_speed': self.avg_typing_speed,
            'max_risk_score': self.max_risk_score,
            'avg_risk_score': self.avg_risk_score,
            'migraine_occurred': self.migraine_occurred
        }


class UserSettings(db.Model):
    """User preferences and calibration data"""
    __tablename__ = 'user_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Monitoring preferences
    enable_keyboard = db.Column(db.Boolean, default=True)
    enable_mouse = db.Column(db.Boolean, default=True)
    enable_webcam = db.Column(db.Boolean, default=True)
    enable_system = db.Column(db.Boolean, default=True)
    
    # Alert thresholds
    alert_threshold_moderate = db.Column(db.Float, default=0.4)
    alert_threshold_high = db.Column(db.Float, default=0.6)
    alert_threshold_critical = db.Column(db.Float, default=0.8)
    
    # Notification settings
    enable_notifications = db.Column(db.Boolean, default=True)
    notification_interval = db.Column(db.Integer, default=30)  # minutes
    
    # Baseline calibration (user's normal values - JSON)
    baseline_data = db.Column(db.Text, default='{}')
    
    # Last updated
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'enable_keyboard': self.enable_keyboard,
            'enable_mouse': self.enable_mouse,
            'enable_webcam': self.enable_webcam,
            'enable_system': self.enable_system,
            'alert_threshold_moderate': self.alert_threshold_moderate,
            'alert_threshold_high': self.alert_threshold_high,
            'alert_threshold_critical': self.alert_threshold_critical,
            'enable_notifications': self.enable_notifications,
            'notification_interval': self.notification_interval,
            'baseline_data': json.loads(self.baseline_data) if self.baseline_data else {},
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
