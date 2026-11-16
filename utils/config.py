"""
Configuration constants for the CATS AI Safety Bot.
"""

import os

# === Risk Classification Thresholds ===
# These thresholds define the boundaries for vehicle status classification.
# They are used both for generating labeled training data and for rule-based explanations.

SPEED_THRESHOLDS = {
    "warning": 100,   # km/h - above this is concerning
    "critical": 140,  # km/h - above this is dangerous
}

ACCELERATION_THRESHOLDS = {
    "warning": 4.0,   # g-force - hard braking/acceleration
    "critical": 7.0,  # g-force - potential collision
}

IMPACT_SCORE_THRESHOLDS = {
    "warning": 0.4,   # 0-1 scale - moderate impact detected
    "critical": 0.7,  # 0-1 scale - severe impact detected
}

BATTERY_THRESHOLDS = {
    "warning": 30,    # % - low battery
    "critical": 10,   # % - critically low battery
}

# GPS status: 1 = active, 0 = lost
GPS_LOSS_IS_WARNING = True

# === Status Labels ===
STATUS_NORMAL = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"

# === Feature Names ===
FEATURE_COLUMNS = [
    "speed",
    "acceleration",
    "impact_score",
    "battery_status",
    "gps_status",
]

TARGET_COLUMN = "risk_label"

# === Model Paths ===
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# === Data Paths ===
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "telemetry_data.csv")

# === Feedback ===
FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feedback_log.csv")

# === UI Settings ===
STATUS_COLORS = {
    STATUS_NORMAL: "#2ecc71",    # Green
    STATUS_WARNING: "#f39c12",   # Orange
    STATUS_CRITICAL: "#e74c3c",  # Red
}

STATUS_EMOJIS = {
    STATUS_NORMAL: "✅",
    STATUS_WARNING: "⚠️",
    STATUS_CRITICAL: "🚨",
}

# === Explanation Templates ===
EXPLANATION_TEMPLATES = {
    "speed_warning": "Vehicle speed ({value:.0f} km/h) exceeds the warning threshold of {threshold} km/h.",
    "speed_critical": "Vehicle speed ({value:.0f} km/h) is dangerously high, exceeding the critical threshold of {threshold} km/h.",
    "acceleration_warning": "Acceleration ({value:.1f} g) indicates hard braking or sudden maneuver.",
    "acceleration_critical": "Extreme acceleration ({value:.1f} g) detected — possible collision or rollover.",
    "impact_warning": "Moderate impact detected (score: {value:.2f}). Vehicle may have hit an obstacle.",
    "impact_critical": "Severe impact detected (score: {value:.2f}). Possible crash — immediate attention needed.",
    "battery_warning": "Battery level is low ({value:.0f}%). Vehicle systems may be affected.",
    "battery_critical": "Battery critically low ({value:.0f}%). Vehicle may lose power imminently.",
    "gps_loss": "GPS signal lost. Vehicle location cannot be determined — potential safety concern.",
}
