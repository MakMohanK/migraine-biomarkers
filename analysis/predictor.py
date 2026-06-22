# ─────────────────────────────────────────────────────────────
#  Migraine Predictor
#  Uses an Isolation Forest (anomaly detection) + a heuristic
#  gradient to flag pre-migraine patterns without labelled data.
# ─────────────────────────────────────────────────────────────

import os
import time
import threading
import numpy as np
import joblib
from typing import Optional   # ← fixes int|None on Python < 3.10

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "model.joblib")

FEATURE_KEYS = [
    "typing_speed_kpm", "mean_iki_ms", "rhythm_cv",
    "mean_hold_ms", "error_rate", "pause_count",
    "mouse_avg_speed_px", "mouse_jitter", "mouse_efficiency",
    "click_rate_per_min", "scroll_rate_per_min", "mouse_idle_sec",
    "blink_rate_bpm", "mean_ear", "head_tilt_deg",
    "head_forward_lean", "face_proximity_px", "squint_ratio",
    "avg_cpu_pct", "avg_mem_pct", "avg_brightness",
    "app_switch_rate", "idle_ratio", "is_late_night",
]


class MigrainePredictor:
    def __init__(self):
        self._lock           = threading.Lock()
        self._model          = None
        self._scaler         = None
        self._history_X      = []
        self._history_y      = []
        self._trained        = False
        self._anomaly_scores = []

    # ── Lifecycle ─────────────────────────────────────────────
    def load_or_init(self):
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        if os.path.exists(MODEL_PATH):
            try:
                bundle        = joblib.load(MODEL_PATH)
                self._model   = bundle["model"]
                self._scaler  = bundle["scaler"]
                self._trained = True
                print("[Predictor] Loaded saved model.")
                return
            except Exception:
                pass
        self._init_fresh_model()

    def _init_fresh_model(self):
        self._scaler = StandardScaler()
        self._model  = IsolationForest(
            n_estimators=150,
            contamination=0.05,
            random_state=42,
        )
        self._trained = False

    def save(self):
        if self._trained:
            joblib.dump({"model": self._model, "scaler": self._scaler}, MODEL_PATH)

    # ── Prediction ────────────────────────────────────────────
    def predict(self, features: dict, risk_score: float) -> dict:
        vec = self._to_vector(features)

        with self._lock:
            self._history_X.append(vec)
            self._history_y.append(1 if risk_score >= 55 else 0)

            # retrain every 50 samples
            if len(self._history_X) % 50 == 0:
                self._retrain()

            if self._trained:
                try:
                    X_sc        = self._scaler.transform([vec])
                    iso_score   = self._model.score_samples(X_sc)[0]
                    anomaly_pct = self._iso_to_pct(iso_score)
                    self._anomaly_scores.append(anomaly_pct)
                    self._anomaly_scores = self._anomaly_scores[-20:]
                    trend = self._trend()
                except Exception:
                    anomaly_pct = risk_score
                    trend       = 0.0
            else:
                anomaly_pct = risk_score
                trend       = 0.0

            blended = max(0.0, min(100.0, risk_score * 0.6 + anomaly_pct * 0.4))

            return {
                "ml_anomaly_pct": round(anomaly_pct, 1),
                "blended_risk":   round(blended,     1),
                "trend":          round(trend,        2),
                "model_trained":  self._trained,
            }

    # ── Retraining ────────────────────────────────────────────
    def _retrain(self):
        X = np.array(self._history_X[-500:])
        try:
            X_sc = self._scaler.fit_transform(X)
            self._model.fit(X_sc)
            self._trained = True
        except Exception as e:
            print(f"[Predictor] Retrain error: {e}")

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _to_vector(features: dict) -> list:
        return [float(features.get(k, 0.0)) for k in FEATURE_KEYS]

    @staticmethod
    def _iso_to_pct(score: float) -> float:
        return ((-max(-0.8, min(0.0, score))) / 0.8) * 100

    def _trend(self) -> float:
        s = self._anomaly_scores[-10:]
        if len(s) < 3:
            return 0.0
        return float(np.polyfit(np.arange(len(s), dtype=float), s, 1)[0])