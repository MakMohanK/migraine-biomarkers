# ─────────────────────────────────────────────────────────────
#  Flask + SocketIO Dashboard Backend v2.0
#  Adds: Helmet Therapy BT endpoints (HC-05 serial)
#  async_mode='threading' — works with real OS threads (pynput safe)
# ─────────────────────────────────────────────────────────────

import os
import sys
import time
import threading
import urllib.request
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, send_file, redirect, request
from flask_socketio import SocketIO, emit

from config import DASHBOARD_HOST, DASHBOARD_PORT

# Optional serial import — graceful fallback if pyserial not installed
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

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

# ── Shared risk state ─────────────────────────────────────────
_state = {
    "risk":       {},
    "prediction": {},
    "features":   {},
    "db":         None,
}

# ── Bluetooth / Serial state ──────────────────────────────────
_bt = {
    "serial":    None,       # active serial.Serial instance
    "port":      None,
    "baud":      9600,
    "connected": False,
    "lock":      threading.Lock(),
}

_server_thread = None


# ─────────────────────────────────────────────────────────────
#  Socket.IO JS auto-download
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  Public API used by monitors
# ─────────────────────────────────────────────────────────────
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
    _bt_disconnect_internal()
    try:
        socketio.stop()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  Risk payload builder
# ─────────────────────────────────────────────────────────────
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
        # ── each domain risk read explicitly so real-time bars always update ──
        "keyboard_risk":  float(risk.get("keyboard_risk",  0)),
        "mouse_risk":     float(risk.get("mouse_risk",     0)),
        "webcam_risk":    float(risk.get("webcam_risk",    0)),
        "system_risk":    float(risk.get("system_risk",    0)),
        "ml_anomaly_pct": prediction.get("ml_anomaly_pct", 0),
        "model_trained":  prediction.get("model_trained",  False),
        "features":       features,
    }


# ─────────────────────────────────────────────────────────────
#  REST — Dashboard
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  REST — Helmet Therapy (HC-05 Bluetooth via Serial)
# ─────────────────────────────────────────────────────────────

@app.route("/api/helmet/ports")
def api_helmet_ports():
    """Scan and return available serial COM ports."""
    if not SERIAL_AVAILABLE:
        return jsonify({"ports": [], "error": "pyserial not installed. Run: pip install pyserial"})
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return jsonify({"ports": ports})
    except Exception as e:
        return jsonify({"ports": [], "error": str(e)})


@app.route("/api/helmet/connect", methods=["POST"])
def api_helmet_connect():
    """Open serial connection to HC-05."""
    if not SERIAL_AVAILABLE:
        return jsonify({"ok": False, "error": "pyserial not installed. Run: pip install pyserial"})

    data = request.get_json(force=True) or {}
    port = data.get("port", "")
    baud = int(data.get("baud", 9600))

    if not port:
        return jsonify({"ok": False, "error": "No port specified"})

    with _bt["lock"]:
        # Close existing connection
        _bt_disconnect_internal()
        try:
            ser = serial.Serial(
                port=port, baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=2,
                write_timeout=2,
            )
            _bt["serial"]    = ser
            _bt["port"]      = port
            _bt["baud"]      = baud
            _bt["connected"] = True
            print(f"[Helmet] Connected to {port} @ {baud}")

            # Start background reader thread
            t = threading.Thread(target=_bt_reader_thread, daemon=True)
            t.start()

            return jsonify({"ok": True, "port": port, "baud": baud})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})


@app.route("/api/helmet/disconnect", methods=["POST"])
def api_helmet_disconnect():
    """Close serial connection."""
    with _bt["lock"]:
        _bt_disconnect_internal()
    return jsonify({"ok": True})


@app.route("/api/helmet/send", methods=["POST"])
def api_helmet_send():
    """
    Send a JSON command string to the device over serial.
    Body: { "payload": "<json string>" }
    Returns any immediate response line read within 1.5 s.
    """
    if not _bt["connected"] or not _bt["serial"]:
        return jsonify({"ok": False, "error": "Not connected"})

    data    = request.get_json(force=True) or {}
    payload = data.get("payload", "")
    if not payload:
        return jsonify({"ok": False, "error": "Empty payload"})

    with _bt["lock"]:
        try:
            ser = _bt["serial"]
            # Send command terminated with \r\n (Arduino expects this)
            ser.write((payload + "\r\n").encode("utf-8"))
            ser.flush()

            # Try to read one response line (up to 1.5 s)
            ser.timeout = 1.5
            response = ""
            try:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    response = line
                    # Forward to all browser clients via socket
                    try:
                        parsed = json.loads(line)
                        socketio.emit("bt_response", parsed)
                    except Exception:
                        socketio.emit("bt_response", {"raw": line})
            except Exception:
                pass

            return jsonify({"ok": True, "response": response})
        except Exception as e:
            _bt["connected"] = False
            return jsonify({"ok": False, "error": str(e)})


@app.route("/api/helmet/status")
def api_helmet_status():
    """Return current BT connection status."""
    return jsonify({
        "connected":        _bt["connected"],
        "port":             _bt["port"],
        "baud":             _bt["baud"],
        "serial_available": SERIAL_AVAILABLE,
    })


# ─────────────────────────────────────────────────────────────
#  WebSocket events
# ─────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("risk_update", _build_payload())
    # Also push BT status on connect
    emit("bt_status", {
        "connected": _bt["connected"],
        "port":      _bt["port"],
    })


@socketio.on("disconnect")
def on_disconnect():
    pass


@socketio.on("bt_scan_ports")
def on_bt_scan_ports():
    """Client requested a port scan via socket."""
    if not SERIAL_AVAILABLE:
        emit("bt_ports", [])
        return
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        emit("bt_ports", ports)
    except Exception:
        emit("bt_ports", [])


# ─────────────────────────────────────────────────────────────
#  BT reader background thread
#  Reads unsolicited lines from Arduino (e.g. session complete)
#  and forwards them to all browser clients via SocketIO
# ─────────────────────────────────────────────────────────────
def _bt_reader_thread():
    print("[Helmet] BT reader thread started")
    while _bt["connected"]:
        try:
            ser = _bt["serial"]
            if not ser or not ser.is_open:
                break
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            print(f"[Helmet RX] {line}")
            try:
                parsed = json.loads(line)
                socketio.emit("bt_response", parsed)
            except Exception:
                socketio.emit("bt_response", {"raw": line})
        except Exception as e:
            print(f"[Helmet] Reader error: {e}")
            break
    _bt["connected"] = False
    print("[Helmet] BT reader thread stopped")
    socketio.emit("bt_response", {"raw": "DISCONNECTED", "status": "DISCONNECTED"})


def _bt_disconnect_internal():
    """Internal helper — must be called with lock held or at shutdown."""
    _bt["connected"] = False
    if _bt["serial"]:
        try:
            _bt["serial"].close()
        except Exception:
            pass
        _bt["serial"] = None
    _bt["port"] = None


# ─────────────────────────────────────────────────────────────
#  Server entry
# ─────────────────────────────────────────────────────────────
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
