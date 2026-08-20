"""
Migraine Predictor - Main Flask Application
A system that monitors laptop activity to predict migraines based on behavioral patterns.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os
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

# Import modules after app initialization to avoid circular imports
from models.database import ActivityLog, MigrainePrediction, UserSession
from monitors.keyboard_monitor import KeyboardMonitor
from monitors.mouse_monitor import MouseMonitor
from monitors.system_monitor import SystemMonitor
from monitors.webcam_monitor import WebcamMonitor
from prediction.migraine_model import MigrainePredictor

# Global monitor instances
keyboard_monitor = None
mouse_monitor = None
system_monitor = None
webcam_monitor = None
predictor = None
monitoring_active = False

# Track how many background cycles have passed (for periodic DB saves)
_bg_cycle_count = 0


def initialize_monitors():
    """Initialize all monitoring modules"""
    global keyboard_monitor, mouse_monitor, system_monitor, webcam_monitor, predictor
    
    keyboard_monitor = KeyboardMonitor()
    mouse_monitor = MouseMonitor()
    system_monitor = SystemMonitor()
    webcam_monitor = WebcamMonitor()
    predictor = MigrainePredictor()
    
    print("✓ All monitors initialized")


def start_monitoring():
    """Start all monitoring modules"""
    global monitoring_active
    
    if monitoring_active:
        return False
    
    monitoring_active = True

    keyboard_monitor.start()
    mouse_monitor.start()
    system_monitor.start()
    webcam_monitor.start()
    
    print("✓ Monitoring started")
    return True


def stop_monitoring():
    """Stop all monitoring modules"""
    global monitoring_active
    
    if not monitoring_active:
        return False
    
    monitoring_active = False
    
    keyboard_monitor.stop()
    mouse_monitor.stop()
    system_monitor.stop()
    webcam_monitor.stop()
    
    print("✓ Monitoring stopped")
    return True


def collect_all_features():
    """Collect features from all monitors"""
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
    return jsonify({
        'success': success,
        'message': 'Monitoring started' if success else 'Monitoring already active'
    })


@app.route('/api/stop', methods=['POST'])
def api_stop_monitoring():
    success = stop_monitoring()
    return jsonify({
        'success': success,
        'message': 'Monitoring stopped' if success else 'Monitoring not active'
    })


@app.route('/api/features')
def get_current_features():
    features = collect_all_features()
    return jsonify(features)


@app.route('/api/predict')
def get_prediction():
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    features = collect_all_features()
    prediction = predictor.predict(features)
    
    return jsonify({
        'prediction': prediction,
        'features': features,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/history')
def get_history():
    limit = request.args.get('limit', 100, type=int)
    
    predictions = MigrainePrediction.query.order_by(
        MigrainePrediction.timestamp.desc()
    ).limit(limit).all()
    
    return jsonify({
        'predictions': [p.to_dict() for p in predictions],
        'count': len(predictions)
    })


@app.route('/api/activity_log')
def get_activity_log():
    hours = request.args.get('hours', 24, type=int)
    
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=hours)
    
    logs = ActivityLog.query.filter(
        ActivityLog.timestamp >= cutoff
    ).order_by(ActivityLog.timestamp.desc()).all()
    
    return jsonify({
        'logs': [log.to_dict() for log in logs],
        'count': len(logs)
    })


@app.route('/api/report_migraine', methods=['POST'])
def report_migraine():
    data = request.json
    
    migraine_type = data.get('type', 'unknown')
    intensity = data.get('intensity', 5)
    notes = data.get('notes', '')

    prediction = MigrainePrediction(
        timestamp=datetime.now(),
        risk_score=1.0,
        risk_level='confirmed',
        migraine_type=migraine_type,
        intensity=intensity,
        notes=notes,
        is_confirmed=True
    )
    
    db.session.add(prediction)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Migraine reported successfully',
        'id': prediction.id
    })


@app.route('/api/therapy_recommendation')
def get_therapy_recommendation():
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    features = collect_all_features()
    prediction = predictor.predict(features)
    recommendations = predictor.get_therapy_recommendations(prediction)
    
    return jsonify({
        'prediction': prediction,
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    })


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    emit('status', {'connected': True, 'monitoring': monitoring_active})
    # Send an immediate data snapshot on connect so the dashboard populates right away
    try:
        features = collect_all_features()
        if predictor:
            prediction = predictor.predict(features)
        else:
            prediction = {'risk_score': 0, 'risk_level': 'low'}
        emit('data_update', {
            'features': features,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error sending initial data on connect: {e}")
    print("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


@socketio.on('request_update')
def handle_update_request():
    """Send current data to requesting client"""
    try:
        features = collect_all_features()

        if predictor:
            prediction = predictor.predict(features)
        else:
            prediction = {'risk_score': 0, 'risk_level': 'low'}

        emit('data_update', {
            'features': features,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error in handle_update_request: {e}")


def background_update():
    """Background task to broadcast periodic updates to all connected clients
       and persist predictions to the database."""
    import time
    global _bg_cycle_count

    while True:
        time.sleep(5)  # Update every 5 seconds
        _bg_cycle_count += 1

        if not monitoring_active:
            continue

        try:
            features = collect_all_features()
            
            if predictor:
                prediction = predictor.predict(features)
            else:
                prediction = {'risk_score': 0, 'risk_level': 'low'}

            # Broadcast to all connected dashboard clients
            socketio.emit('data_update', {
                'features': features,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            })

            # ── Persist to database ───────────────────────────────────────
            with app.app_context():
                # 1. Always write an ActivityLog entry
                log = ActivityLog(
                    timestamp=datetime.now(),
                    keyboard_features=json.dumps(features.get('keyboard', {})),
                    mouse_features=json.dumps(features.get('mouse', {})),
                    system_features=json.dumps(features.get('system', {})),
                    webcam_features=json.dumps(features.get('webcam', {})),
                    risk_score=prediction['risk_score']
                )
                db.session.add(log)

                # 2. Write a MigrainePrediction record every 60 s (every 12 cycles)
                #    so the History page always has live data to display.
                if _bg_cycle_count % 12 == 0 or prediction['risk_score'] >= 0.3:
                    pred_record = MigrainePrediction(
                        timestamp=datetime.now(),
                        risk_score=prediction['risk_score'],
                        risk_level=prediction.get('risk_level', 'low'),
                        migraine_type=prediction.get('migraine_type'),
                        contributing_factors=json.dumps(
                            prediction.get('contributing_factors', [])
                        ),
                        therapy_recommendations=json.dumps([]),
                        is_confirmed=False
                    )
                    db.session.add(pred_record)

                db.session.commit()

        except Exception as e:
            print(f"Background update error: {e}")


# ============== Main Entry Point ==============

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("✓ Database initialized")
    
    # Initialize monitors
    initialize_monitors()

    # Auto-start monitoring so data flows immediately on launch
    start_monitoring()

    # Start background broadcast / persistence thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()

    print("\n" + "=" * 50)
    print("  Migraine Predictor System")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)