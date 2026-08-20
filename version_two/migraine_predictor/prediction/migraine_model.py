"""
Migraine Prediction Model
Uses collected behavioral features to predict migraine risk and type.

This module implements:
1. Feature extraction and normalization
2. Risk score calculation
3. Migraine type classification
4. Therapy recommendations
"""

import json
import os
from datetime import datetime, timedelta
from collections import deque
import statistics

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Using rule-based prediction.")


class MigrainePredictor:
    """Predicts migraine risk based on behavioral features"""
    
    def __init__(self, model_path=None):
        """
        Initialize the migraine predictor.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model_path = model_path
        self.risk_model = None
        self.type_model = None
        self.scaler = None
        
        # Feature history for trend analysis
        self.feature_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=50)
        
        # Feature weights for rule-based prediction
        self.feature_weights = {
            'keyboard_fatigue_score': 0.20,
            'mouse_fatigue_score': 0.15,
            'webcam_fatigue_score': 0.25,
            'system_fatigue_score': 0.15,
            'eye_strain_score': 0.25
        }
        
        # Migraine type indicators
        self.type_indicators = {
            'tension': {
                'indicators': ['head_tilt', 'typing_errors', 'neck_strain'],
                'description': 'Tension-type headache',
                'common_triggers': ['stress', 'poor_posture', 'screen_time']
            },
            'migraine_with_aura': {
                'indicators': ['eye_strain', 'light_sensitivity', 'visual_fatigue'],
                'description': 'Migraine with visual aura',
                'common_triggers': ['bright_light', 'screen_glare', 'eye_strain']
            },
            'migraine_without_aura': {
                'indicators': ['fatigue', 'cognitive_slowdown', 'concentration_loss'],
                'description': 'Migraine without aura',
                'common_triggers': ['fatigue', 'stress', 'sleep_disruption']
            },
            'cluster': {
                'indicators': ['late_night_usage', 'irregular_patterns', 'high_stress'],
                'description': 'Cluster-type headache',
                'common_triggers': ['sleep_changes', 'alcohol', 'stress']
            }
        }
        
        # Therapy recommendations
        self.therapy_options = {
            'pressure_therapy': {
                'description': 'Apply gentle pressure to temples and forehead',
                'intensity_range': (1, 5),
                'duration_minutes': 15,
                'suitable_for': ['tension', 'migraine_without_aura']
            },
            'cold_therapy': {
                'description': 'Apply cold compress to forehead and neck',
                'temperature_range': (10, 15),  # Celsius
                'duration_minutes': 20,
                'suitable_for': ['migraine_with_aura', 'migraine_without_aura']
            },
            'warm_therapy': {
                'description': 'Apply warmth to neck and shoulders',
                'temperature_range': (38, 42),  # Celsius
                'duration_minutes': 15,
                'suitable_for': ['tension', 'cluster']
            },
            'rest_recommendation': {
                'description': 'Take a break from screen',
                'duration_minutes': 30,
                'suitable_for': ['all']
            },
            'hydration_reminder': {
                'description': 'Drink water - dehydration can trigger migraines',
                'duration_minutes': 5,
                'suitable_for': ['all']
            },
            'dim_lights': {
                'description': 'Reduce screen brightness and ambient lighting',
                'duration_minutes': 20,
                'suitable_for': ['migraine_with_aura', 'eye_strain']
            }
        }
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        elif SKLEARN_AVAILABLE:
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize a default model structure"""
        self.scaler = StandardScaler()
        # Models will be trained when enough data is available
        self.risk_model = None
        self.type_model = None
    
    def _load_model(self, path):
        """Load pre-trained model from disk"""
        try:
            model_data = joblib.load(path)
            self.risk_model = model_data.get('risk_model')
            self.type_model = model_data.get('type_model')
            self.scaler = model_data.get('scaler')
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, path):
        """Save trained model to disk"""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            model_data = {
                'risk_model': self.risk_model,
                'type_model': self.type_model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, path)
            print(f"Model saved to {path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def extract_features(self, raw_features):
        """
        Extract and normalize features from raw monitor data.
        
        Args:
            raw_features: Dict containing features from all monitors
            
        Returns:
            Dict of normalized features for prediction
        """
        extracted = {}
        
        # Keyboard features
        kb = raw_features.get('keyboard', {})
        extracted['typing_speed'] = kb.get('typing_speed', 0)
        extracted['typing_speed_deviation'] = kb.get('speed_deviation', 0)
        extracted['error_rate'] = kb.get('error_rate', 0)
        extracted['error_deviation'] = kb.get('error_deviation', 0)
        extracted['avg_interval'] = kb.get('avg_interval', 0)
        extracted['rhythm_consistency'] = kb.get('rhythm_consistency', 0)
        extracted['pause_count'] = kb.get('pause_count', 0)
        extracted['keyboard_fatigue'] = kb.get('keyboard_fatigue_score', 0)
        
        # Mouse features
        ms = raw_features.get('mouse', {})
        extracted['mouse_speed'] = ms.get('avg_speed', 0)
        extracted['mouse_speed_deviation'] = ms.get('speed_deviation', 0)
        extracted['click_rate'] = ms.get('click_rate', 0)
        extracted['idle_time'] = ms.get('total_idle_time', 0)
        extracted['jitter_count'] = ms.get('jitter_count', 0)
        extracted['mouse_fatigue'] = ms.get('mouse_fatigue_score', 0)
        
        # Webcam features
        wc = raw_features.get('webcam', {})
        extracted['blink_rate'] = wc.get('blink_rate', 0)
        extracted['blink_rate_deviation'] = wc.get('blink_rate_deviation', 0)
        extracted['eye_openness'] = wc.get('avg_eye_openness', 0)
        extracted['face_distance'] = wc.get('avg_face_distance', 0)
        extracted['head_tilt'] = abs(wc.get('avg_head_roll', 0))
        extracted['eye_strain'] = wc.get('eye_strain_score', 0)
        extracted['webcam_fatigue'] = wc.get('webcam_fatigue_score', 0)
        
        # System features
        sys = raw_features.get('system', {})
        extracted['cpu_usage'] = sys.get('avg_cpu', 0)
        extracted['memory_usage'] = sys.get('avg_memory', 0)
        extracted['app_switch_rate'] = sys.get('app_switch_rate', 0)
        extracted['session_duration'] = sys.get('session_duration_minutes', 0)
        extracted['is_late_night'] = 1 if sys.get('is_late_night', False) else 0
        extracted['workload'] = sys.get('workload_score', 0)
        extracted['system_fatigue'] = sys.get('system_fatigue_score', 0)
        
        return extracted
    
    def predict(self, raw_features):
        """
        Predict migraine risk from collected features.
        
        Args:
            raw_features: Dict containing features from all monitors
            
        Returns:
            Dict containing prediction results
        """
        # Extract features
        features = self.extract_features(raw_features)
        
        # Store in history for trend analysis
        features['timestamp'] = datetime.now().isoformat()
        self.feature_history.append(features)
        
        # Calculate risk score
        if SKLEARN_AVAILABLE and self.risk_model:
            risk_score = self._ml_predict_risk(features)
        else:
            risk_score = self._rule_based_predict_risk(features)
        
        # Determine risk level
        risk_level = self._get_risk_level(risk_score)
        
        # Predict migraine type if risk is elevated
        migraine_type = None
        type_confidence = 0
        if risk_score > 0.4:
            migraine_type, type_confidence = self._predict_type(features)
        
        # Calculate trend
        trend = self._calculate_trend()
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(features)
        
        # Build prediction result
        prediction = {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'migraine_type': migraine_type,
            'type_confidence': round(type_confidence, 2) if type_confidence else None,
            'trend': trend,
            'contributing_factors': contributing_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def _rule_based_predict_risk(self, features):
        """
        Rule-based risk prediction when ML model not available.
        
        Combines fatigue scores from all monitors with weights.
        """
        risk_score = 0.0
        
        # Weighted combination of fatigue scores
        risk_score += self.feature_weights['keyboard_fatigue_score'] * features.get('keyboard_fatigue', 0)
        risk_score += self.feature_weights['mouse_fatigue_score'] * features.get('mouse_fatigue', 0)
        risk_score += self.feature_weights['webcam_fatigue_score'] * features.get('webcam_fatigue', 0)
        risk_score += self.feature_weights['system_fatigue_score'] * features.get('system_fatigue', 0)
        risk_score += self.feature_weights['eye_strain_score'] * features.get('eye_strain', 0)
        
        # Additional risk factors
        
        # Long session increases risk
        session_hours = features.get('session_duration', 0) / 60
        if session_hours > 2:
            risk_score += 0.1 * min(1, (session_hours - 2) / 4)
        
        # Late night usage increases risk
        if features.get('is_late_night', 0):
            risk_score += 0.15
        
        # High error rate indicates cognitive issues
        if features.get('error_deviation', 0) > 0.5:
            risk_score += 0.1 * min(1, features['error_deviation'])
        
        # Reduced blink rate indicates eye strain
        if features.get('blink_rate_deviation', 0) < -0.3:
            risk_score += 0.1 * min(1, abs(features['blink_rate_deviation']))
        
        return min(1.0, risk_score)
    
    def _ml_predict_risk(self, features):
        """ML-based risk prediction"""
        try:
            feature_vector = self._features_to_vector(features)
            if self.scaler:
                feature_vector = self.scaler.transform([feature_vector])
            
            # Get probability of high risk
            risk_prob = self.risk_model.predict_proba(feature_vector)[0][1]
            return risk_prob
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._rule_based_predict_risk(features)
    
    def _features_to_vector(self, features):
        """Convert feature dict to vector for ML model"""
        feature_keys = [
            'typing_speed', 'error_rate', 'rhythm_consistency', 'keyboard_fatigue',
            'mouse_speed', 'click_rate', 'idle_time', 'jitter_count', 'mouse_fatigue',
            'blink_rate', 'eye_openness', 'face_distance', 'head_tilt', 'eye_strain', 'webcam_fatigue',
            'cpu_usage', 'memory_usage', 'app_switch_rate', 'session_duration', 'is_late_night', 'system_fatigue'
        ]
        return [features.get(k, 0) for k in feature_keys]
    
    def _get_risk_level(self, risk_score):
        """Convert risk score to risk level category"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.5:
            return 'moderate'
        elif risk_score < 0.7:
            return 'high'
        else:
            return 'critical'
    
    def _predict_type(self, features):
        """
        Predict the most likely migraine type based on feature patterns.
        
        Returns:
            Tuple of (migraine_type, confidence)
        """
        type_scores = {}
        
        # Tension-type indicators
        tension_score = 0
        if features.get('head_tilt', 0) > 10:
            tension_score += 0.3
        if features.get('error_deviation', 0) > 0.3:
            tension_score += 0.2
        if features.get('system_fatigue', 0) > 0.5:
            tension_score += 0.2
        if features.get('session_duration', 0) > 120:
            tension_score += 0.3
        type_scores['tension'] = tension_score
        
        # Migraine with aura indicators
        aura_score = 0
        if features.get('eye_strain', 0) > 0.5:
            aura_score += 0.4
        if features.get('blink_rate_deviation', 0) < -0.3:
            aura_score += 0.3
        if features.get('face_distance', 0) < 40:
            aura_score += 0.3
        type_scores['migraine_with_aura'] = aura_score
        
        # Migraine without aura indicators
        no_aura_score = 0
        if features.get('keyboard_fatigue', 0) > 0.5:
            no_aura_score += 0.3
        if features.get('typing_speed_deviation', 0) < -0.3:
            no_aura_score += 0.3
        if features.get('workload', 0) > 0.6:
            no_aura_score += 0.2
        if features.get('pause_count', 0) > 5:
            no_aura_score += 0.2
        type_scores['migraine_without_aura'] = no_aura_score
        
        # Cluster indicators
        cluster_score = 0
        if features.get('is_late_night', 0):
            cluster_score += 0.4
        if features.get('workload', 0) > 0.7:
            cluster_score += 0.3
        if features.get('app_switch_rate', 0) > 5:
            cluster_score += 0.3
        type_scores['cluster'] = cluster_score
        
        # Find the type with highest score
        if not type_scores:
            return None, 0
        
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]
        
        # Only return if confidence is reasonable
        if confidence > 0.3:
            return best_type, confidence
        
        return None, 0
    
    def _calculate_trend(self):
        """
        Calculate risk trend from recent predictions.
        
        Returns:
            'increasing', 'stable', or 'decreasing'
        """
        if len(self.prediction_history) < 3:
            return 'stable'
        
        recent = list(self.prediction_history)[-10:]
        scores = [p['risk_score'] for p in recent]
        
        # Calculate trend using simple linear regression
        n = len(scores)
        if n < 3:
            return 'stable'
        
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.02:
            return 'increasing'
        elif slope < -0.02:
            return 'decreasing'
        return 'stable'
    
    def _identify_contributing_factors(self, features):
        """
        Identify the main factors contributing to migraine risk.
        
        Returns:
            List of contributing factors with their impact
        """
        factors = []
        
        # Check each factor
        if features.get('keyboard_fatigue', 0) > 0.4:
            factors.append({
                'factor': 'Typing fatigue',
                'impact': 'high' if features['keyboard_fatigue'] > 0.6 else 'moderate',
                'description': 'Reduced typing speed and increased errors'
            })
        
        if features.get('eye_strain', 0) > 0.4:
            factors.append({
                'factor': 'Eye strain',
                'impact': 'high' if features['eye_strain'] > 0.6 else 'moderate',
                'description': 'Reduced blink rate and eye fatigue'
            })
        
        if features.get('webcam_fatigue', 0) > 0.4:
            factors.append({
                'factor': 'Visual fatigue',
                'impact': 'high' if features['webcam_fatigue'] > 0.6 else 'moderate',
                'description': 'Face position and head tilt indicate strain'
            })
        
        if features.get('session_duration', 0) > 120:
            hours = features['session_duration'] / 60
            factors.append({
                'factor': 'Extended screen time',
                'impact': 'high' if hours > 3 else 'moderate',
                'description': f'Working for {hours:.1f} hours without break'
            })
        
        if features.get('is_late_night', 0):
            factors.append({
                'factor': 'Late night usage',
                'impact': 'moderate',
                'description': 'Using computer during late night hours'
            })
        
        if features.get('workload', 0) > 0.5:
            factors.append({
                'factor': 'High workload',
                'impact': 'high' if features['workload'] > 0.7 else 'moderate',
                'description': 'High CPU/memory usage and multitasking'
            })
        
        if features.get('head_tilt', 0) > 15:
            factors.append({
                'factor': 'Poor posture',
                'impact': 'moderate',
                'description': 'Head tilted more than 15 degrees'
            })
        
        # Sort by impact
        impact_order = {'high': 0, 'moderate': 1, 'low': 2}
        factors.sort(key=lambda x: impact_order.get(x['impact'], 2))
        
        return factors[:5]  # Return top 5 factors
    
    def get_therapy_recommendations(self, prediction):
        """
        Get therapy recommendations based on prediction.
        
        Args:
            prediction: Prediction result dict
            
        Returns:
            List of therapy recommendations
        """
        recommendations = []
        risk_score = prediction.get('risk_score', 0)
        risk_level = prediction.get('risk_level', 'low')
        migraine_type = prediction.get('migraine_type')
        contributing_factors = prediction.get('contributing_factors', [])
        
        # Always recommend breaks for moderate+ risk
        if risk_score > 0.3:
            recommendations.append({
                'therapy': 'rest_recommendation',
                'priority': 'high' if risk_score > 0.6 else 'medium',
                **self.therapy_options['rest_recommendation']
            })
        
        # Hydration reminder
        if risk_score > 0.2:
            recommendations.append({
                'therapy': 'hydration_reminder',
                'priority': 'medium',
                **self.therapy_options['hydration_reminder']
            })
        
        # Type-specific recommendations
        if migraine_type:
            # Cold therapy for migraines with aura
            if migraine_type == 'migraine_with_aura':
                recommendations.append({
                    'therapy': 'cold_therapy',
                    'priority': 'high',
                    **self.therapy_options['cold_therapy']
                })
                recommendations.append({
                    'therapy': 'dim_lights',
                    'priority': 'high',
                    **self.therapy_options['dim_lights']
                })
            
            # Pressure therapy for tension headaches
            elif migraine_type == 'tension':
                recommendations.append({
                    'therapy': 'pressure_therapy',
                    'priority': 'high',
                    **self.therapy_options['pressure_therapy']
                })
                recommendations.append({
                    'therapy': 'warm_therapy',
                    'priority': 'medium',
                    **self.therapy_options['warm_therapy']
                })
            
            # Cold therapy for migraines without aura
            elif migraine_type == 'migraine_without_aura':
                recommendations.append({
                    'therapy': 'cold_therapy',
                    'priority': 'high',
                    **self.therapy_options['cold_therapy']
                })
            
            # Warm therapy for cluster
            elif migraine_type == 'cluster':
                recommendations.append({
                    'therapy': 'warm_therapy',
                    'priority': 'high',
                    **self.therapy_options['warm_therapy']
                })
        
        # Factor-specific recommendations
        for factor in contributing_factors:
            if factor['factor'] == 'Eye strain':
                if not any(r['therapy'] == 'dim_lights' for r in recommendations):
                    recommendations.append({
                        'therapy': 'dim_lights',
                        'priority': 'medium',
                        **self.therapy_options['dim_lights']
                    })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
        
        return recommendations
    
    def train_model(self, training_data, labels):
        """
        Train the prediction model on collected data.
        
        Args:
            training_data: List of feature dicts
            labels: List of outcomes (0=no migraine, 1=migraine)
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available for training")
            return False
        
        if len(training_data) < 50:
            print("Need at least 50 samples for training")
            return False
        
        try:
            # Convert to feature vectors
            X = [self._features_to_vector(f) for f in training_data]
            y = labels
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train risk prediction model
            self.risk_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.risk_model.fit(X_scaled, y)
            
            print("Model trained successfully")
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def get_statistics(self):
        """Get prediction statistics"""
        if not self.prediction_history:
            return {}
        
        recent = list(self.prediction_history)
        scores = [p['risk_score'] for p in recent]
        
        return {
            'total_predictions': len(recent),
            'avg_risk_score': statistics.mean(scores),
            'max_risk_score': max(scores),
            'min_risk_score': min(scores),
            'high_risk_count': sum(1 for p in recent if p['risk_level'] in ['high', 'critical']),
            'current_trend': self._calculate_trend()
        }
