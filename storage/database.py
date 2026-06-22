
# ─────────────────────────────────────────────────────────────
#  SQLite Database Layer
#  Stores feature snapshots, risk scores and alert history
# ─────────────────────────────────────────────────────────────

import os
import json
import sqlite3
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "migraine_monitor.db")


class Database:
    def __init__(self, path: str = DB_PATH):
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────
    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL    NOT NULL,
                    source      TEXT    NOT NULL,
                    features    TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_scores (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       REAL    NOT NULL,
                    composite_risk  REAL,
                    keyboard_risk   REAL,
                    mouse_risk      REAL,
                    webcam_risk     REAL,
                    system_risk     REAL,
                    risk_level      TEXT,
                    ml_anomaly_pct  REAL,
                    blended_risk    REAL,
                    trend           REAL,
                    eta_minutes     INTEGER
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL    NOT NULL,
                    level       TEXT    NOT NULL,
                    message     TEXT    NOT NULL,
                    risk_score  REAL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time  REAL    NOT NULL,
                    end_time    REAL,
                    max_risk    REAL    DEFAULT 0,
                    alert_count INTEGER DEFAULT 0
                );
            """)

    # ── Feature Snapshots ─────────────────────────────────────
    def insert_snapshot(self, source: str, features: dict):
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO feature_snapshots (timestamp, source, features) VALUES (?,?,?)",
                (time.time(), source, json.dumps(features))
            )

    def get_snapshots(self, source: str = None, limit: int = 200) -> list:
        with self._lock, self._connect() as conn:
            if source:
                rows = conn.execute(
                    "SELECT * FROM feature_snapshots WHERE source=? ORDER BY timestamp DESC LIMIT ?",
                    (source, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feature_snapshots ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    # ── Risk Scores ───────────────────────────────────────────
    def insert_risk(self, risk: dict, prediction: dict, eta_minutes):
        with self._lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO risk_scores
                    (timestamp, composite_risk, keyboard_risk, mouse_risk,
                     webcam_risk, system_risk, risk_level, ml_anomaly_pct,
                     blended_risk, trend, eta_minutes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                time.time(),
                risk.get("composite_risk"),
                risk.get("keyboard_risk"),
                risk.get("mouse_risk"),
                risk.get("webcam_risk"),
                risk.get("system_risk"),
                risk.get("risk_level"),
                prediction.get("ml_anomaly_pct"),
                prediction.get("blended_risk"),
                prediction.get("trend"),
                eta_minutes,
            ))

    def get_risk_history(self, limit: int = 100) -> list:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM risk_scores ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_risk_summary(self) -> dict:
        with self._lock, self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)            AS total,
                    AVG(blended_risk)   AS avg_risk,
                    MAX(blended_risk)   AS max_risk,
                    MIN(blended_risk)   AS min_risk
                FROM risk_scores
                WHERE timestamp > ?
            """, (time.time() - 86400,)).fetchone()
        return dict(row) if row else {}

    # ── Alerts ────────────────────────────────────────────────
    def insert_alert(self, level: str, message: str, risk_score: float):
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO alerts (timestamp, level, message, risk_score) VALUES (?,?,?,?)",
                (time.time(), level, message, risk_score)
            )

    def get_alerts(self, limit: int = 50) -> list:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Sessions ──────────────────────────────────────────────
    def start_session(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (start_time) VALUES (?)",
                (time.time(),)
            )
            return cur.lastrowid

    def end_session(self, session_id: int, max_risk: float, alert_count: int):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET end_time=?, max_risk=?, alert_count=? WHERE id=?",
                (time.time(), max_risk, alert_count, session_id)
            )

    # ── Helpers ───────────────────────────────────────────────
    def _connect(self):
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_snapshot(row) -> dict:
        d = dict(row)
        try:
            d["features"] = json.loads(d["features"])
        except Exception:
            pass
        return d
