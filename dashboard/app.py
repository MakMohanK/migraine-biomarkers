# ─────────────────────────────────────────────────────────────
#  Flask + SocketIO Dashboard Backend
#  async_mode='threading' — works with real OS threads (pynput safe)
# ─────────────────────────────────────────────────────────────

import os
import sys
import time
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, send_file, redirect
from flask_socketio import SocketIO, emit

from config import DASHBOARD_HOST, DASHBOARD_PORT

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_JS = os.path.join(BASE_DIR, "dashboard", "static", "js")
SIO_LOCAL = os.path.join(STATIC_JS, "socket.io.min.js")
SIO_CDN   = "https://cdn.socket.io/4.7.2/socket.io.min.js"
SIO_CDN2  = "https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"

app = Flask(__name__)
app.config["SECRET_KEY"] = "migraine-sense-secret-2024"

# ── async_mode='threading' uses real OS threads — pynput compatible ──
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",       # ← was 'eventlet' which broke pynput
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# ── Shared state ──────────────────────────────────────────────
_state = {
    "risk":       {},
    "prediction": {},
    "features":   {},
    "db":         None,
}

_server_thread = None


# ── Auto-download socket.io.min.js ────────────────────────────
def _ensure_socketio_js():
    if os.path.exists(SIO_LOCAL) and os.path.getsize(SIO_LOCAL) > 10_000:
        return
    os.makedirs(STATIC_JS, exist_ok=True)
    for url in (SIO_CDN, SIO_CDN2):
        try:
            print(f"[Dashboard] Downloading socket.io.min.js …")
            urllib.request.urlretrieve(url, SIO_LOCAL)
            print("[Dashboard] socket.io.min.js saved locally ✓")
            return
        except Exception as e:
            print(f"[Dashboard] CDN failed ({e}), trying next …")
    print("[Dashboard] WARNING: socket.io.min.js unavailable — browser uses CDN fallback.")


# ── Safe route: /assets/socket.io.js (never conflicts with /socket.io/) ──
@app.route("/assets/socket.io.js")
def serve_socketio_js():
    if os.path.exists(SIO_LOCAL) and os.path.getsize(SIO_LOCAL) > 10_000:
        return send_file(SIO_LOCAL, mimetype="application/javascript")
    return redirect(SIO_CDN)


# ── Public API ────────────────────────────────────────────────
def inject_state(risk, prediction, features, alerts, db):
    _state.update({"risk": risk, "prediction": prediction,
                   "features": features, "db": db})


def push_update(risk: dict, prediction: dict, features: dict):
    _state["risk"]       = risk
    _state["prediction"] = prediction
    _state["features"]   = features
    try:
        socketio.emit("risk_update", _build_payload())
    except Exception:
        pass


def stop_dashboard():
    """Unblock run_dashboard() cleanly."""
    try:
        socketio.stop()
    except Exception:
        pass


# ── REST endpoints ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/current")
def api_current():
    return jsonify(_build_payload())


@app.route("/api/history")
def api_history():
    db = _state.get("db")
    return jsonify(db.get_risk_history(limit=120) if db else [])


@app.route("/api/alerts")
def api_alerts():
    db = _state.get("db")
    return jsonify(db.get_alerts(limit=50) if db else [])


@app.route("/api/summary")
def api_summary():
    db = _state.get("db")
    return jsonify(db.get_risk_summary() if db else {})


# ── WebSocket ─────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("risk_update", _build_payload())


@socketio.on("disconnect")
def on_disconnect():
    pass


# ── Payload builder ───────────────────────────────────────────
def _build_payload() -> dict:
    risk       = _state.get("risk",       {})
    prediction = _state.get("prediction", {})
    features   = _state.get("features",   {})
    return {
        "timestamp":      time.time(),
        "composite_risk": risk.get("composite_risk",       0),
        "blended_risk":   prediction.get("blended_risk",   0),
        "risk_level":     risk.get("risk_level",           "LOW"),
        "eta_minutes":    risk.get("eta_minutes",          None),
        "trend":          prediction.get("trend",          0),
        "keyboard_risk":  risk.get("keyboard_risk",        0),
        "mouse_risk":     risk.get("mouse_risk",           0),
        "webcam_risk":    risk.get("webcam_risk",          0),
        "system_risk":    risk.get("system_risk",          0),
        "ml_anomaly_pct": prediction.get("ml_anomaly_pct", 0),
        "model_trained":  prediction.get("model_trained",  False),
        "features":       features,
    }


# ── Server entry ──────────────────────────────────────────────
def run_dashboard():
    _ensure_socketio_js()
    socketio.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,   # needed for threading mode
        log_output=False,
    )