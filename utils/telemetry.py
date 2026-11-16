"""
Telemetry data processing utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import (
    FEATURE_COLUMNS, STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL,
    SPEED_THRESHOLDS, ACCELERATION_THRESHOLDS, IMPACT_SCORE_THRESHOLDS,
    BATTERY_THRESHOLDS, EXPLANATION_TEMPLATES
)


def load_telemetry(filepath: str) -> pd.DataFrame:
    """Load telemetry data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


def get_latest_reading(df: pd.DataFrame, vehicle_id: Optional[str] = None) -> pd.Series:
    """Get the most recent telemetry reading, optionally filtered by vehicle."""
    if vehicle_id:
        df = df[df["vehicle_id"] == vehicle_id]
    return df.sort_values("timestamp").iloc[-1]


def get_vehicle_ids(df: pd.DataFrame) -> List[str]:
    """Get list of unique vehicle IDs."""
    return sorted(df["vehicle_id"].unique().tolist())


def get_vehicle_history(df: pd.DataFrame, vehicle_id: str, n: int = 10) -> pd.DataFrame:
    """Get the last N readings for a vehicle."""
    vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp")
    return vehicle_df.tail(n)


def compute_statistics(df: pd.DataFrame, vehicle_id: Optional[str] = None) -> Dict:
    """Compute summary statistics for telemetry data."""
    if vehicle_id:
        df = df[df["vehicle_id"] == vehicle_id]

    stats = {}
    for col in FEATURE_COLUMNS:
        if col == "gps_status":
            stats[col] = {
                "active_pct": (df[col] == 1).mean() * 100,
                "loss_count": (df[col] == 0).sum(),
            }
        else:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
            }
    
    # Risk distribution
    if "risk_label" in df.columns:
        risk_counts = df["risk_label"].value_counts().to_dict()
        stats["risk_distribution"] = risk_counts

    return stats


def generate_explanation(row: pd.Series) -> List[str]:
    """
    Generate human-readable explanations for why a specific reading
    received its risk classification.
    """
    explanations = []

    # Speed checks
    speed = row["speed"]
    if speed >= SPEED_THRESHOLDS["critical"]:
        explanations.append(
            EXPLANATION_TEMPLATES["speed_critical"].format(
                value=speed, threshold=SPEED_THRESHOLDS["critical"]
            )
        )
    elif speed >= SPEED_THRESHOLDS["warning"]:
        explanations.append(
            EXPLANATION_TEMPLATES["speed_warning"].format(
                value=speed, threshold=SPEED_THRESHOLDS["warning"]
            )
        )

    # Acceleration checks
    accel = row["acceleration"]
    if accel >= ACCELERATION_THRESHOLDS["critical"]:
        explanations.append(
            EXPLANATION_TEMPLATES["acceleration_critical"].format(
                value=accel, threshold=ACCELERATION_THRESHOLDS["critical"]
            )
        )
    elif accel >= ACCELERATION_THRESHOLDS["warning"]:
        explanations.append(
            EXPLANATION_TEMPLATES["acceleration_warning"].format(
                value=accel, threshold=ACCELERATION_THRESHOLDS["warning"]
            )
        )

    # Impact checks
    impact = row["impact_score"]
    if impact >= IMPACT_SCORE_THRESHOLDS["critical"]:
        explanations.append(
            EXPLANATION_TEMPLATES["impact_critical"].format(
                value=impact, threshold=IMPACT_SCORE_THRESHOLDS["critical"]
            )
        )
    elif impact >= IMPACT_SCORE_THRESHOLDS["warning"]:
        explanations.append(
            EXPLANATION_TEMPLATES["impact_warning"].format(
                value=impact, threshold=IMPACT_SCORE_THRESHOLDS["warning"]
            )
        )

    # Battery checks
    battery = row["battery_status"]
    if battery <= BATTERY_THRESHOLDS["critical"]:
        explanations.append(
            EXPLANATION_TEMPLATES["battery_critical"].format(
                value=battery, threshold=BATTERY_THRESHOLDS["critical"]
            )
        )
    elif battery <= BATTERY_THRESHOLDS["warning"]:
        explanations.append(
            EXPLANATION_TEMPLATES["battery_warning"].format(
                value=battery, threshold=BATTERY_THRESHOLDS["warning"]
            )
        )

    # GPS check
    if row["gps_status"] == 0:
        explanations.append(EXPLANATION_TEMPLATES["gps_loss"])

    if not explanations:
        explanations.append("All telemetry readings are within normal operating parameters.")

    return explanations
