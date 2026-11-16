"""
Generate simulated vehicle telemetry data for the CATS AI Safety Bot.

Produces a realistic dataset with:
- Normal driving patterns (majority of data)
- Warning scenarios (aggressive driving, low battery, etc.)
- Critical scenarios (crashes, extreme speeds, GPS loss during emergencies)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import (
    STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL,
    SPEED_THRESHOLDS, ACCELERATION_THRESHOLDS,
    IMPACT_SCORE_THRESHOLDS, BATTERY_THRESHOLDS,
)


def classify_reading(speed, acceleration, impact_score, battery_status, gps_status):
    """Classify a telemetry reading based on thresholds."""
    # CRITICAL conditions (any one triggers CRITICAL)
    if (speed >= SPEED_THRESHOLDS["critical"] or
        acceleration >= ACCELERATION_THRESHOLDS["critical"] or
        impact_score >= IMPACT_SCORE_THRESHOLDS["critical"] or
        battery_status <= BATTERY_THRESHOLDS["critical"]):
        return STATUS_CRITICAL

    # WARNING conditions
    warning_count = 0
    if speed >= SPEED_THRESHOLDS["warning"]:
        warning_count += 1
    if acceleration >= ACCELERATION_THRESHOLDS["warning"]:
        warning_count += 1
    if impact_score >= IMPACT_SCORE_THRESHOLDS["warning"]:
        warning_count += 1
    if battery_status <= BATTERY_THRESHOLDS["warning"]:
        warning_count += 1
    if gps_status == 0:
        warning_count += 1

    # Two or more warning-level flags → CRITICAL
    if warning_count >= 2:
        return STATUS_CRITICAL

    if warning_count >= 1:
        return STATUS_WARNING

    return STATUS_NORMAL


def generate_normal_driving(n_samples, rng):
    """Generate normal driving telemetry."""
    return {
        "speed": rng.normal(60, 20, n_samples).clip(0, 95),
        "acceleration": rng.exponential(0.8, n_samples).clip(0, 3.5),
        "impact_score": rng.exponential(0.05, n_samples).clip(0, 0.35),
        "battery_status": rng.normal(75, 15, n_samples).clip(35, 100),
        "gps_status": rng.choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], n_samples),
    }


def generate_warning_scenarios(n_samples, rng):
    """Generate warning-level telemetry."""
    speed = np.concatenate([
        rng.normal(110, 10, n_samples // 3).clip(100, 135),  # High speed
        rng.normal(70, 20, n_samples // 3).clip(0, 95),       # Normal speed but other issues
        rng.normal(90, 15, n_samples - 2 * (n_samples // 3)).clip(60, 135),  # Moderate speed
    ])
    rng.shuffle(speed)

    acceleration = np.concatenate([
        rng.normal(5.0, 1.0, n_samples // 3).clip(4.0, 6.5),  # Hard braking
        rng.exponential(1.0, n_samples - (n_samples // 3)).clip(0, 3.5),  # Normal
    ])
    rng.shuffle(acceleration)

    impact_score = np.concatenate([
        rng.normal(0.5, 0.1, n_samples // 4).clip(0.4, 0.65),  # Moderate impact
        rng.exponential(0.1, n_samples - (n_samples // 4)).clip(0, 0.35),  # Low impact
    ])
    rng.shuffle(impact_score)

    battery_status = np.concatenate([
        rng.normal(22, 5, n_samples // 4).clip(12, 30),  # Low battery
        rng.normal(65, 15, n_samples - (n_samples // 4)).clip(35, 100),  # Normal
    ])
    rng.shuffle(battery_status)

    gps_status = rng.choice([1, 1, 1, 1, 0], n_samples)

    return {
        "speed": speed,
        "acceleration": acceleration,
        "impact_score": impact_score,
        "battery_status": battery_status,
        "gps_status": gps_status,
    }


def generate_critical_scenarios(n_samples, rng):
    """Generate critical-level telemetry."""
    speed = np.concatenate([
        rng.normal(155, 15, n_samples // 3).clip(140, 220),  # Extreme speed
        rng.normal(80, 30, n_samples - (n_samples // 3)).clip(0, 130),  # Varied
    ])
    rng.shuffle(speed)

    acceleration = np.concatenate([
        rng.normal(8.5, 1.5, n_samples // 3).clip(7.0, 15.0),  # Collision-level
        rng.normal(3.0, 2.0, n_samples - (n_samples // 3)).clip(0, 6.0),  # Varied
    ])
    rng.shuffle(acceleration)

    impact_score = np.concatenate([
        rng.normal(0.85, 0.1, n_samples // 3).clip(0.7, 1.0),  # Severe impact
        rng.normal(0.3, 0.2, n_samples - (n_samples // 3)).clip(0, 0.65),  # Varied
    ])
    rng.shuffle(impact_score)

    battery_status = np.concatenate([
        rng.normal(5, 3, n_samples // 4).clip(0, 10),  # Critically low
        rng.normal(50, 20, n_samples - (n_samples // 4)).clip(15, 100),  # Varied
    ])
    rng.shuffle(battery_status)

    gps_status = rng.choice([1, 1, 0, 0, 0], n_samples)

    return {
        "speed": speed,
        "acceleration": acceleration,
        "impact_score": impact_score,
        "battery_status": battery_status,
        "gps_status": gps_status,
    }


def generate_dataset(n_total=8000, seed=42):
    """
    Generate the full simulated telemetry dataset.
    
    Distribution: ~60% NORMAL, ~25% WARNING, ~15% CRITICAL
    """
    rng = np.random.default_rng(seed)

    n_normal = int(n_total * 0.60)
    n_warning = int(n_total * 0.25)
    n_critical = n_total - n_normal - n_warning

    print(f"Generating {n_total} telemetry readings...")
    print(f"  NORMAL:   {n_normal}")
    print(f"  WARNING:  {n_warning}")
    print(f"  CRITICAL: {n_critical}")

    # Generate scenarios
    normal_data = generate_normal_driving(n_normal, rng)
    warning_data = generate_warning_scenarios(n_warning, rng)
    critical_data = generate_critical_scenarios(n_critical, rng)

    # Combine all data
    all_data = {}
    for key in ["speed", "acceleration", "impact_score", "battery_status", "gps_status"]:
        all_data[key] = np.concatenate([
            normal_data[key], warning_data[key], critical_data[key]
        ])

    n = len(all_data["speed"])

    # Generate timestamps (over 30 days, multiple readings per day)
    base_time = datetime(2026, 10, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=int(rng.integers(0, 43200))) for _ in range(n)]
    timestamps.sort()

    # Generate vehicle IDs and realistic GPS coordinates (corridors around Bangalore/Outer Ring Road)
    vehicle_ids = [f"VH-{rng.integers(1001, 1006):04d}" for _ in range(n)]

    # Base coordinates for vehicles
    corridor_centers = {
        "VH-1001": (12.9716, 77.5946),  # Central
        "VH-1002": (12.9279, 77.6271),  # Koramangala
        "VH-1003": (12.9850, 77.7300),  # Whitefield
        "VH-1004": (13.0358, 77.5970),  # Hebbal / Airport Expressway
        "VH-1005": (12.8452, 77.6602),  # Electronic City
    }

    latitudes = []
    longitudes = []
    for vid in vehicle_ids:
        base_lat, base_lon = corridor_centers[vid]
        # Realistic offset within ~8-12 km radius
        lat_offset = float(rng.normal(0, 0.035))
        lon_offset = float(rng.normal(0, 0.035))
        latitudes.append(np.round(base_lat + lat_offset, 5))
        longitudes.append(np.round(base_lon + lon_offset, 5))

    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "vehicle_id": vehicle_ids,
        "speed": np.round(all_data["speed"], 1),
        "acceleration": np.round(all_data["acceleration"], 2),
        "impact_score": np.round(all_data["impact_score"], 3),
        "battery_status": np.round(all_data["battery_status"], 1),
        "gps_status": all_data["gps_status"].astype(int),
        "latitude": latitudes,
        "longitude": longitudes,
    })

    # Classify each reading
    df["risk_label"] = df.apply(
        lambda row: classify_reading(
            row["speed"], row["acceleration"],
            row["impact_score"], row["battery_status"],
            row["gps_status"]
        ), axis=1
    )

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main():
    """Generate and save the telemetry dataset."""
    df = generate_dataset()

    # Save
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "telemetry_data.csv")
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRisk label distribution:")
    print(df["risk_label"].value_counts().to_string())
    print(f"\nFeature statistics:")
    print(df[["speed", "acceleration", "impact_score", "battery_status"]].describe().round(2).to_string())
    print(f"\nGPS status: {(df['gps_status'] == 1).mean()*100:.1f}% active")
    print(f"\nSample rows:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
