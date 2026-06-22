# ─────────────────────────────────────────────────────────────
#  Risk Calculator
#  Converts raw features → 0-100 risk score per domain,
#  then blends them into a single MigraineRisk index.
# ─────────────────────────────────────────────────────────────

from typing import Optional          # ← Python 3.9 compatible
from config import FEATURE_WEIGHTS


# ── Normalisation helpers ─────────────────────────────────────
def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def _scale(value, bad_threshold, good_threshold=0.0, invert=False):
    """
    Map value linearly between good_threshold and bad_threshold → 0..100.
    invert=True  → higher value is BETTER (e.g. typing speed).
    """
    if invert:
        value, good_threshold, bad_threshold = (
            bad_threshold - value,
            0,
            bad_threshold - good_threshold,
        )
    if bad_threshold == good_threshold:
        return 0.0
    ratio = (value - good_threshold) / (bad_threshold - good_threshold)
    return _clamp(ratio * 100)


# ─────────────────────────────────────────────────────────────
class RiskCalculator:

    # ── Domain scorers ────────────────────────────────────────
    @staticmethod
    def keyboard_risk(f: dict) -> float:
        """
        High risk signals:
          - very slow typing (< 20 kpm)
          - high error rate (> 15 %)
          - long inter-keystroke intervals
          - many pauses
          - long key holds (tremor / fatigue)
          - high rhythm variance
        """
        score = 0.0

        speed = f.get("typing_speed_kpm", 60)
        score += _scale(speed, bad_threshold=10, good_threshold=80, invert=True) * 0.20

        err = f.get("error_rate", 0) * 100          # → percent
        score += _scale(err, bad_threshold=20, good_threshold=0) * 0.25

        iki = f.get("mean_iki_ms", 200)
        score += _scale(iki, bad_threshold=800, good_threshold=100) * 0.15

        pauses = f.get("pause_count", 0)
        score += _scale(pauses, bad_threshold=20, good_threshold=0) * 0.15

        hold = f.get("mean_hold_ms", 80)
        score += _scale(hold, bad_threshold=300, good_threshold=50) * 0.15

        cv = f.get("rhythm_cv", 0)
        score += _scale(cv, bad_threshold=1.5, good_threshold=0) * 0.10

        return _clamp(score)

    @staticmethod
    def mouse_risk(f: dict) -> float:
        """
        High risk signals:
          - very slow / erratic movement
          - high jitter (trembling)
          - low efficiency (wandering cursor)
          - long idle stretches
          - high or low click rate (both indicate distress)
        """
        score = 0.0

        speed = f.get("mouse_avg_speed_px", 300)
        score += _scale(speed, bad_threshold=20, good_threshold=500, invert=True) * 0.15

        jitter = f.get("mouse_jitter", 0)
        score += _scale(jitter, bad_threshold=0.6, good_threshold=0.0) * 0.25

        eff = f.get("mouse_efficiency", 1.0) * 100
        score += _scale(eff, bad_threshold=30, good_threshold=90, invert=True) * 0.15

        idle = f.get("mouse_idle_sec", 0)
        score += _scale(idle, bad_threshold=300, good_threshold=10) * 0.20

        cr = f.get("click_rate_per_min", 5)
        # Abnormally high OR low click rate → risk
        if cr > 40:
            score += 20
        elif cr < 1:
            score += 15

        scroll = f.get("scroll_rate_per_min", 5)
        score += _scale(scroll, bad_threshold=60, good_threshold=0) * 0.10

        ici = f.get("mean_ici_ms", 500)
        score += _scale(ici, bad_threshold=3000, good_threshold=200) * 0.15

        return _clamp(score)

    @staticmethod
    def webcam_risk(f: dict) -> float:
        """
        High risk signals:
          - blink rate too low (< 8) or too high (> 25)
          - low EAR → squinting / eye strain
          - head tilted or leaning forward
          - face very close to screen
          - high squint ratio
          - no face (user away / slumped)
        """
        score = 0.0

        bpm = f.get("blink_rate_bpm", 15)
        if bpm < 8:
            score += 30
        elif bpm > 25:
            score += 20
        else:
            score += _scale(abs(bpm - 15), bad_threshold=10, good_threshold=0) * 0.15

        ear = f.get("mean_ear", 0.28)
        score += _scale(ear, bad_threshold=0.15, good_threshold=0.30, invert=True) * 0.20

        tilt = f.get("head_tilt_deg", 0)
        score += _scale(abs(tilt), bad_threshold=25, good_threshold=3) * 0.15

        fwd = f.get("head_forward_lean", 0)
        score += _scale(fwd, bad_threshold=0.05, good_threshold=0.0) * 0.10

        prox = f.get("face_proximity_px", 120)
        score += _scale(prox, bad_threshold=200, good_threshold=80) * 0.15

        squint = f.get("squint_ratio", 0) * 100
        score += _scale(squint, bad_threshold=40, good_threshold=0) * 0.15

        no_face = f.get("no_face_ratio", 0) * 100
        score += _scale(no_face, bad_threshold=60, good_threshold=5) * 0.10

        return _clamp(score)

    @staticmethod
    def system_risk(f: dict) -> float:
        """
        High risk signals:
          - very high CPU / memory load
          - very bright screen
          - rapid app switching (cognitive overload)
          - high idle ratio (disengaged)
          - late night usage
          - low battery (stress)
        """
        score = 0.0

        cpu = f.get("avg_cpu_pct", 30)
        score += _scale(cpu, bad_threshold=90, good_threshold=20) * 0.15

        mem = f.get("avg_mem_pct", 50)
        score += _scale(mem, bad_threshold=90, good_threshold=30) * 0.10

        bright = f.get("avg_brightness", 50)
        score += _scale(bright, bad_threshold=100, good_threshold=40) * 0.20

        sw = f.get("app_switch_rate", 2)
        score += _scale(sw, bad_threshold=20, good_threshold=1) * 0.20

        idle = f.get("idle_ratio", 0) * 100
        score += _scale(idle, bad_threshold=70, good_threshold=5) * 0.10

        if f.get("is_late_night", 0):
            score += 20

        bat = f.get("battery_pct", 100)
        if bat < 20 and not f.get("battery_charging", 1):
            score += 10

        return _clamp(score)

    # ── Composite score ───────────────────────────────────────
    def compute(self, features: dict) -> dict:
        kb_score  = self.keyboard_risk(features)
        ms_score  = self.mouse_risk(features)
        wc_score  = self.webcam_risk(features)
        sys_score = self.system_risk(features)

        w = FEATURE_WEIGHTS
        composite = (
            kb_score  * w["keyboard"] +
            ms_score  * w["mouse"]    +
            wc_score  * w["webcam"]   +
            sys_score * w["system"]
        )

        level = (
            "CRITICAL" if composite >= 75 else
            "HIGH"     if composite >= 55 else
            "MODERATE" if composite >= 30 else
            "LOW"
        )

        eta_minutes = self._estimate_eta(composite)

        return {
            "composite_risk":  round(composite,  1),
            "keyboard_risk":   round(kb_score,   1),
            "mouse_risk":      round(ms_score,   1),
            "webcam_risk":     round(wc_score,   1),
            "system_risk":     round(sys_score,  1),
            "risk_level":      level,
            "eta_minutes":     eta_minutes,
        }

    @staticmethod
    def _estimate_eta(score: float) -> Optional[int]:
        """Rough estimate of minutes until migraine onset based on score."""
        if score < 30:
            return None
        elif score < 45:
            return 180
        elif score < 55:
            return 120
        elif score < 65:
            return 90
        elif score < 75:
            return 60
        else:
            return 30
