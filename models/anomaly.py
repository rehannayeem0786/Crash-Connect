"""
Anomaly Detection module for the CATS AI Safety Bot.

Uses Isolation Forest for unsupervised anomaly detection
to catch unusual telemetry patterns the classifier might miss.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import FEATURE_COLUMNS, DATASET_PATH

ANOMALY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "anomaly_detector.pkl")
ANOMALY_SCALER_PATH = os.path.join(os.path.dirname(__file__), "anomaly_scaler.pkl")


def train_anomaly_detector():
    """
    Train Isolation Forest on NORMAL data only.
    Anomalies are readings that deviate from normal driving patterns.
    """
    print("Training Anomaly Detector (Isolation Forest)...")

    df = pd.read_csv(DATASET_PATH)

    # Train only on NORMAL data — anomalies are deviations from normal
    normal_df = df[df["risk_label"] == "NORMAL"]
    X_normal = normal_df[FEATURE_COLUMNS].values

    print(f"  Training on {len(normal_df)} NORMAL readings")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect ~5% anomalies in normal data
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)

    # Save
    joblib.dump(iso_forest, ANOMALY_MODEL_PATH)
    joblib.dump(scaler, ANOMALY_SCALER_PATH)

    # Test on full dataset
    X_all = df[FEATURE_COLUMNS].values
    X_all_scaled = scaler.transform(X_all)
    predictions = iso_forest.predict(X_all_scaled)
    scores = iso_forest.decision_function(X_all_scaled)

    n_anomalies = (predictions == -1).sum()
    print(f"  Anomalies detected in full dataset: {n_anomalies}/{len(df)} ({n_anomalies/len(df)*100:.1f}%)")

    # Breakdown by risk label
    df["is_anomaly"] = predictions == -1
    for label in ["NORMAL", "WARNING", "CRITICAL"]:
        subset = df[df["risk_label"] == label]
        anom_count = subset["is_anomaly"].sum()
        print(f"  {label}: {anom_count}/{len(subset)} anomalies ({anom_count/len(subset)*100:.1f}%)")

    print(f"\n  Model saved to {ANOMALY_MODEL_PATH}")
    print("✅ Anomaly detector trained!")

    return iso_forest, scaler


def detect_anomaly(features, iso_forest=None, scaler=None):
    """
    Check if a single reading is anomalous.
    Returns: (is_anomaly: bool, anomaly_score: float)
    """
    if iso_forest is None:
        iso_forest = joblib.load(ANOMALY_MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(ANOMALY_SCALER_PATH)

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = iso_forest.predict(X_scaled)[0]
    score = iso_forest.decision_function(X_scaled)[0]

    return prediction == -1, float(score)


if __name__ == "__main__":
    train_anomaly_detector()
