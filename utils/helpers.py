# ─────────────────────────────────────────────────────────────
#  Utility Helpers
# ─────────────────────────────────────────────────────────────

import datetime


def format_timestamp(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def risk_color(level: str) -> str:
    return {
        "LOW":      "#22c55e",
        "MODERATE": "#eab308",
        "HIGH":     "#f97316",
        "CRITICAL": "#ef4444",
    }.get(level, "#6b7280")


def eta_string(eta_minutes) -> str:
    if eta_minutes is None:
        return "No risk detected"
    if eta_minutes >= 120:
        h = eta_minutes // 60
        return f"~{h} hours"
    return f"~{eta_minutes} minutes"


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
