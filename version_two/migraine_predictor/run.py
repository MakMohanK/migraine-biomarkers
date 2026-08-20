"""
Migraine Predictor - Run Script
Start the Flask application with all monitoring modules.
"""

import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import threading

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'migraine-predictor-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///migraine_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)


# ============== Database Models ==============

class ActivityLog(db.Model):
    __tablename__ = 'activity_logs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    keyboard_features = db.Column(db.Text, default='{}')
    mouse_features = db.Column(db.Text, default='{}')
    system_features = db.Column(db.Text, default='{}')
    webcam_features = db.Column(db.Text, default='{}')
    risk_score = db.Column(db.Float, default=0.0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'risk_score': self.risk_score
        }


class MigrainePrediction(db.Model):
    __tablename__ = 'migraine_predictions'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    risk_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(20), default='low')
    migraine_type = db.Column(db.String(50), nullable=True)
    intensity = db.Column(db.Integer, nullable=True)
    is_confirmed = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)
    contributing_factors = db.Column(db.Text, default='{}')
    
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
            'contributing_factors': json.loads(self.contributing_factors) if self.contributing_factors else {}
        }


# ============== Import Monitors ==============

from monitors.keyboard_monitor import KeyboardMonitor
from monitors.mouse_monitor import MouseMonitor
from monitors.system_monitor import SystemMonitor
from monitors.webcam_monitor import WebcamMonitor
from prediction.migraine_model import MigrainePredictor

# Global instances
keyboard_monitor = None
mouse_monitor = None
system_monitor = None
webcam_monitor = None
predictor = None
monitoring_active = False


def initialize_monitors():
    global keyboard_monitor, mouse_monitor, system_monitor, webcam_monitor, predictor
    keyboard_monitor = KeyboardMonitor()
    mouse_monitor = MouseMonitor()
    system_monitor = SystemMonitor()
    webcam_monitor = WebcamMonitor()
    predictor = MigrainePredictor()
    print("[OK] All monitors initialized")


def start_monitoring():
    global monitoring_active
    if monitoring_active:
        return False
    monitoring_active = True
    keyboard_monitor.start()
    mouse_monitor.start()
    system_monitor.start()
    webcam_monitor.start()
    print("[OK] Monitoring started")
    return True


def stop_monitoring():
    global monitoring_active
    if not monitoring_active:
        return False
    monitoring_active = False
    keyboard_monitor.stop()
    mouse_monitor.stop()
    system_monitor.stop()
    webcam_monitor.stop()
    print("[OK] Monitoring stopped")
    return True


def collect_all_features():
    features = {}
    if keyboard_monitor:
        features['keyboard'] = keyboard_monitor.get_features()
    if mouse_monitor:
        features['mouse'] = mouse_monitor.get_features()
    if system_monitor:
        features['system'] = system_monitor.get_features()
    if webcam_monitor:
        features['webcam'] = webcam_monitor.get_features()
    features['timestamp'] = datetime.now().isoformat()
    return features


# ============== Routes ==============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')


# ============== API Endpoints ==============

@app.route('/api/status')
def get_status():
    return jsonify({
        'monitoring_active': monitoring_active,
        'monitors': {
            'keyboard': keyboard_monitor.is_running() if keyboard_monitor else False,
            'mouse': mouse_monitor.is_running() if mouse_monitor else False,
            'system': system_monitor.is_running() if system_monitor else False,
            'webcam': webcam_monitor.is_running() if webcam_monitor else False
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start', methods=['POST'])
def api_start_monitoring():
    success = start_monitoring()
    return jsonify({'success': success, 'message': 'Monitoring started' if success else 'Already active'})

@app.route('/api/stop', methods=['POST'])
def api_stop_monitoring():
    success = stop_monitoring()
    return jsonify({'success': success, 'message': 'Monitoring stopped' if success else 'Not active'})

@app.route('/api/features')
def get_current_features():
    return jsonify(collect_all_features())

@app.route('/api/predict')
def get_prediction():
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    features = collect_all_features()
    prediction = predictor.predict(features)
    return jsonify({'prediction': prediction, 'features': features, 'timestamp': datetime.now().isoformat()})

@app.route('/api/history')
def get_history():
    limit = request.args.get('limit', 100, type=int)
    predictions = MigrainePrediction.query.order_by(MigrainePrediction.timestamp.desc()).limit(limit).all()
    return jsonify({'predictions': [p.to_dict() for p in predictions], 'count': len(predictions)})

@app.route('/api/report_migraine', methods=['POST'])
def report_migraine():
    data = request.json
    prediction = MigrainePrediction(
        timestamp=datetime.now(),
        risk_score=1.0,
        risk_level='confirmed',
        migraine_type=data.get('type', 'unknown'),
        intensity=data.get('intensity', 5),
        notes=data.get('notes', ''),
        is_confirmed=True
    )
    db.session.add(prediction)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Migraine reported', 'id': prediction.id})

@app.route('/api/therapy_suggestion', methods=['POST'])
def get_therapy_suggestion():
    """
    Accepts pain survey answers + optional risk_score_pct.
    Returns therapy recommendation from mapping table in plain language.
    """
    data = request.json or {}

    # Determine risk score percentage
    risk_score_pct = data.get('risk_score_pct', None)
    if risk_score_pct is None:
        if predictor:
            features = collect_all_features()
            prediction = predictor.predict(features)
            risk_score_pct = round(prediction['risk_score'] * 100)
        else:
            risk_score_pct = 0

    risk_score_pct = max(0, min(100, int(risk_score_pct)))
    answers = data.get('answers', {})

    # ── Mapping table ──────────────────────────────────────────────────────
    THERAPY_MAP = [
        (0,  30, 'Low',      None,         None,  None),
        (31, 35, 'Moderate', 'FCM + BCM',  'LKL', 'low'),
        (36, 40, 'Moderate', 'FCM + BCM',  'ACL', 'low'),
        (41, 45, 'Moderate', 'FHM + BHM',  'SPL', 'low'),
        (46, 50, 'Moderate', 'FHM + BHM',  'LKM', 'medium'),
        (51, 55, 'High',     'FHM + BHM',  'ACM', 'medium'),
        (56, 60, 'High',     'FHM + BHM',  'SPM', 'medium'),
        (61, 65, 'High',     'FHH + BHH',  'LKH', 'high'),
        (66, 70, 'High',     'FHH + BHH',  'ACH', 'high'),
        (71, 75, 'Critical', 'FCM + BCM',  'SPH', 'high'),
        (76, 80, 'Critical', 'FCH + BCH',  'LKH', 'high'),
        (81, 85, 'Critical', 'FCH + BCH',  'ACH', 'high'),
        (86, 90, 'Critical', 'FCH + BCH',  'SPH', 'high'),
        (91, 95, 'Critical', 'FCH + BCH',  'ACH', 'high'),
        (96,100, 'Critical', 'FCH + BCH',  'SPH', 'high'),
    ]

    # ── Plain language descriptions ────────────────────────────────────────
    HEAD_DESCRIPTIONS = {
        'FCM': {'title': 'Cold Pack — Front of Head',              'icon': '❄️', 'steps': [
            'Wrap a cold pack or a bag of ice in a thin cloth.',
            'Place it on your forehead and temples.',
            'Keep it there for 15 minutes at a comfortable medium pressure.',
            'Take a 5-minute break, then repeat if needed.'
        ]},
        'BCM': {'title': 'Cold Pack — Back of Head & Neck',        'icon': '❄️', 'steps': [
            'Wrap a cold pack or a bag of ice in a thin cloth.',
            'Place it on the back of your head and upper neck.',
            'Keep it there for 15 minutes.',
            'Rest in a quiet, dimly lit room during this time.'
        ]},
        'FCH': {'title': 'Cold Pack — Front of Head (Intense)',    'icon': '🧊', 'steps': [
            'Use a gel cold pack kept in the freezer for at least 30 minutes.',
            'Wrap in a cloth and press firmly on your forehead and temples.',
            'Hold for 20 minutes. Do not apply directly to bare skin.',
            'Lie down flat if possible and keep your eyes closed.'
        ]},
        'BCH': {'title': 'Cold Pack — Back of Head & Neck (Intense)', 'icon': '🧊', 'steps': [
            'Use a gel cold pack kept in the freezer for at least 30 minutes.',
            'Wrap in a cloth and press firmly on the base of your skull and neck.',
            'Hold for 20 minutes while lying down in a dark room.',
            'Breathe slowly and deeply throughout.'
        ]},
        'FHM': {'title': 'Warm Pack — Temples & Forehead',         'icon': '🌡️', 'steps': [
            'Warm a heat pack in the microwave or use a warm damp towel.',
            'Place it over your forehead and temples.',
            'Keep it there for 15 minutes at a comfortable warmth.',
            'Close your eyes and breathe slowly.'
        ]},
        'BHM': {'title': 'Warm Pack — Back of Head & Neck',        'icon': '🌡️', 'steps': [
            'Warm a heat pack or use a warm damp towel.',
            'Place it on the back of your head and neck muscles.',
            'Keep it there for 15 minutes to relax muscle tension.',
            'Sit or lie in a comfortable, quiet position.'
        ]},
        'FHH': {'title': 'Warm Pack — Temples & Forehead (Intense)', 'icon': '🔥', 'steps': [
            'Use a microwavable heat pack heated to a comfortably warm level.',
            'Press firmly over your forehead and temples.',
            'Hold for 20 minutes — check skin every 5 minutes to avoid overheating.',
            'Dim the lights and avoid screen use during this time.'
        ]},
        'BHH': {'title': 'Warm Pack — Back of Head & Neck (Intense)', 'icon': '🔥', 'steps': [
            'Use a microwavable heat pack heated to a comfortably warm level.',
            'Press firmly on the back of your neck and base of skull.',
            'Hold for 20 minutes to release deep muscle tension.',
            'Lie face-down or rest your head on a soft surface.'
        ]},
    }

    PRESSURE_DESCRIPTIONS = {
        'LKL': {'title': 'Light Kneading Massage',               'icon': '👐', 'steps': [
            'Use your fingertips to gently massage your temples in small circles.',
            'Work across the top of your scalp with very light pressure.',
            'Continue for 5 minutes — it should feel soothing, not painful.',
        ]},
        'ACL': {'title': 'Gentle Acupressure — LI4 Point',       'icon': '🤲', 'steps': [
            'Find the LI4 point: the fleshy web between your thumb and index finger.',
            'Pinch this point gently with the thumb and index finger of your other hand.',
            'Hold for 3 minutes with light, steady pressure. Then switch hands.',
            'This point is well known for relieving head and face pain.',
        ]},
        'SPL': {'title': 'Light Spot Pressure — GB20 Points',    'icon': '👆', 'steps': [
            'Find the GB20 points: two hollows at the base of your skull, either side of your neck.',
            'Place both thumbs on these points.',
            'Apply gentle upward pressure for 3 minutes while breathing deeply.',
        ]},
        'LKM': {'title': 'Medium Kneading Massage',              'icon': '👐', 'steps': [
            'Use your fingertips to massage your temples, scalp, and the back of your neck.',
            'Apply medium pressure in slow circular motions.',
            'Continue for 8 minutes — firm but comfortable.',
        ]},
        'ACM': {'title': 'Medium Acupressure — LI4 & P6 Points', 'icon': '🤲', 'steps': [
            'Apply medium pressure to the LI4 point (thumb-index web) for 5 minutes per hand.',
            'Find the P6 point: 3 finger-widths above the inner wrist crease, between the two tendons.',
            'Press P6 firmly with your thumb for 5 minutes per wrist.',
        ]},
        'SPM': {'title': 'Medium Spot Pressure — GB20 & GV20',   'icon': '👆', 'steps': [
            'Apply medium pressure to GB20 (base of skull hollows) with both thumbs for 5 minutes.',
            'Find GV20: the very crown (top) of your head.',
            'Press GV20 with your middle finger for 5 minutes while sitting upright.',
        ]},
        'LKH': {'title': 'Firm Kneading Massage',                'icon': '💪', 'steps': [
            'Use your thumbs and fingers to firmly knead the back of your neck muscles.',
            'Work up to your temples and scalp with firm circular pressure.',
            'Continue for 10 minutes — this should feel like a deep massage.',
            'Tilt your head gently to each side while massaging.',
        ]},
        'ACH': {'title': 'Firm Acupressure — LI4, P6 & ST36',   'icon': '🤲', 'steps': [
            'Apply firm pressure to LI4 (thumb-index web) for 7 minutes per hand.',
            'Apply firm pressure to P6 (inner wrist, 3 fingers up) for 7 minutes per wrist.',
            'Find ST36: 4 finger-widths below your kneecap, outside the shin bone.',
            'Press ST36 firmly for 7 minutes per leg. This helps reduce severe pain.',
        ]},
        'SPH': {'title': 'Firm Spot Pressure — GB20, GV20 & BL2', 'icon': '👆', 'steps': [
            'Apply firm pressure to GB20 (base of skull) with both thumbs for 7 minutes.',
            'Press GV20 (crown of head) firmly with your middle finger for 7 minutes.',
            'Find BL2: the inner edge of each eyebrow at the brow ridge.',
            'Press BL2 on both sides firmly for 7 minutes. Breathe deeply throughout.',
        ]},
    }

    # ── Find matching row ──────────────────────────────────────────────────
    matched_row = (96, 100, 'Critical', 'FCH + BCH', 'SPH', 'high')  # fallback
    for row in THERAPY_MAP:
        lo, hi, level, head, pressure, intensity = row
        if lo <= risk_score_pct <= hi:
            matched_row = row
            break

    lo, hi, risk_level, head_code, pressure_code, intensity_level = matched_row

    # ── Calculate recommended session duration ─────────────────────────────
    def get_session_duration(risk_lvl, intensity_lvl):
        if intensity_lvl is None:
            return None
        if risk_lvl == 'Critical':
            return 60
        if risk_lvl == 'High' and intensity_lvl == 'high':
            return 45
        if intensity_lvl == 'medium':
            return 30
        return 20  # low intensity / Moderate

    session_duration_minutes = get_session_duration(risk_level, intensity_level)

    if head_code is None:
        return jsonify({
            'risk_score_pct': risk_score_pct,
            'risk_level': 'Low',
            'no_therapy_needed': True,
            'message': (
                "Your migraine risk is currently low — no therapy is needed right now! "
                "Keep drinking water, take a short break every 45 minutes, "
                "and ensure your screen brightness is comfortable. 😊"
            )
        })

    # Build therapy detail objects from codes
    head_codes = [c.strip() for c in head_code.split('+')]
    head_details = [HEAD_DESCRIPTIONS[c] for c in head_codes if c in HEAD_DESCRIPTIONS]
    pressure_detail = PRESSURE_DESCRIPTIONS.get(pressure_code, {})

    # Personalise head therapy based on pain location answer
    pain_location = answers.get('pain_location', 'both')
    if pain_location == 'front' and len(head_details) > 1:
        head_details = [head_details[0]]   # front pack only
    elif pain_location == 'back' and len(head_details) > 1:
        head_details = [head_details[1]]   # back pack only

    return jsonify({
        'risk_score_pct': risk_score_pct,
        'risk_level': risk_level,
        'head_therapy_code': head_code,
        'pressure_therapy_code': pressure_code,
        'intensity_level': intensity_level,
        'session_duration_minutes': session_duration_minutes,
        'no_therapy_needed': False,
        'head_therapy': head_details,
        'pressure_therapy': pressure_detail,
        'answers': answers
    })


@app.route('/api/therapy_recommendation')
def get_therapy_recommendation():
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    features = collect_all_features()
    prediction = predictor.predict(features)
    recommendations = predictor.get_therapy_recommendations(prediction)
    return jsonify({'prediction': prediction, 'recommendations': recommendations})


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    # Immediately tell the new client that monitoring is active
    emit('status', {'connected': True, 'monitoring': monitoring_active})

    # Push a live prediction straight away so the dashboard isn't blank
    if monitoring_active and predictor:
        try:
            features   = collect_all_features()
            prediction = predictor.predict(features)
            emit('data_update', {
                'features':   features,
                'prediction': prediction,
                'timestamp':  datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Connect push error: {e}")

@socketio.on('request_update')
def handle_update_request():
    features = collect_all_features()
    if predictor and monitoring_active:
        prediction = predictor.predict(features)
    else:
        prediction = {'risk_score': 0, 'risk_level': 'unknown', 'trend': 'stable', 'contributing_factors': []}
    emit('data_update', {'features': features, 'prediction': prediction, 'timestamp': datetime.now().isoformat()})


def background_update():
    import time
    while True:
        if monitoring_active:
            try:
                features = collect_all_features()
                if predictor:
                    prediction = predictor.predict(features)
                else:
                    prediction = {'risk_score': 0, 'risk_level': 'low',
                                  'trend': 'stable', 'contributing_factors': []}

                # ── Save every prediction to DB (not just high-risk) ────────
                with app.app_context():
                    # Save activity log
                    log = ActivityLog(
                        timestamp=datetime.now(),
                        keyboard_features=json.dumps(features.get('keyboard', {})),
                        mouse_features=json.dumps(features.get('mouse', {})),
                        system_features=json.dumps(features.get('system', {})),
                        webcam_features=json.dumps(features.get('webcam', {})),
                        risk_score=prediction['risk_score']
                    )
                    db.session.add(log)

                    # Save prediction record
                    pred_record = MigrainePrediction(
                        timestamp=datetime.now(),
                        risk_score=prediction['risk_score'],
                        risk_level=prediction.get('risk_level', 'low'),
                        migraine_type=prediction.get('migraine_type'),
                        contributing_factors=json.dumps(
                            prediction.get('contributing_factors', [])
                        )
                    )
                    db.session.add(pred_record)
                    db.session.commit()

                # ── Broadcast to all connected clients ───────────────────────
                socketio.emit('data_update', {
                    'features':   features,
                    'prediction': prediction,
                    'timestamp':  datetime.now().isoformat()
                })

            except Exception as e:
                print(f"Background update error: {e}")

        time.sleep(5)   # emit every 5 seconds


# ============== Main ==============

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("[OK] Database initialized")

    initialize_monitors()

    # ── Auto-start all 4 monitors immediately on launch ──────────────────
    start_monitoring()

    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()

    print("\n" + "="*50)
    print("  Migraine Predictor System")
    print("  Monitoring: ACTIVE (all 4 streams running)")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
