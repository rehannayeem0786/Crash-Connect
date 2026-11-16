"""
Tests for the CATS AI Safety Bot risk classifier.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import (
    FEATURE_COLUMNS, STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL,
    CLASSIFIER_PATH, SCALER_PATH, DATASET_PATH,
)
from utils.telemetry import generate_explanation


class TestDataset:
    """Tests for the generated dataset."""

    def test_dataset_exists(self):
        assert os.path.exists(DATASET_PATH), "Dataset file not found"

    def test_dataset_has_rows(self):
        df = pd.read_csv(DATASET_PATH)
        assert len(df) > 0, "Dataset is empty"

    def test_dataset_columns(self):
        df = pd.read_csv(DATASET_PATH)
        expected = FEATURE_COLUMNS + ["timestamp", "vehicle_id", "risk_label"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_risk_labels_valid(self):
        df = pd.read_csv(DATASET_PATH)
        valid_labels = {STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL}
        actual_labels = set(df["risk_label"].unique())
        assert actual_labels.issubset(valid_labels), f"Invalid labels: {actual_labels - valid_labels}"

    def test_feature_ranges(self):
        df = pd.read_csv(DATASET_PATH)
        assert df["speed"].min() >= 0, "Speed cannot be negative"
        assert df["acceleration"].min() >= 0, "Acceleration cannot be negative"
        assert df["impact_score"].min() >= 0, "Impact score cannot be negative"
        assert df["impact_score"].max() <= 1.0, "Impact score cannot exceed 1.0"
        assert df["battery_status"].min() >= 0, "Battery cannot be negative"
        assert df["battery_status"].max() <= 100, "Battery cannot exceed 100%"
        assert set(df["gps_status"].unique()).issubset({0, 1}), "GPS must be 0 or 1"


class TestModel:
    """Tests for the trained model."""

    def test_model_exists(self):
        assert os.path.exists(CLASSIFIER_PATH), "Model file not found"
        assert os.path.exists(SCALER_PATH), "Scaler file not found"

    def test_model_loads(self):
        import joblib
        model_data = joblib.load(CLASSIFIER_PATH)
        assert "model" in model_data, "Model missing from saved file"
        assert "label_encoder" in model_data, "Label encoder missing"

    def test_normal_prediction(self):
        """Normal driving should predict NORMAL."""
        import joblib
        model_data = joblib.load(CLASSIFIER_PATH)
        scaler = joblib.load(SCALER_PATH)
        clf = model_data["model"]
        le = model_data["label_encoder"]

        # Normal: low speed, low accel, no impact, full battery, GPS active
        X = np.array([[50, 0.5, 0.02, 90, 1]])
        X_scaled = scaler.transform(X)
        pred = le.inverse_transform(clf.predict(X_scaled))[0]
        assert pred == STATUS_NORMAL, f"Expected NORMAL, got {pred}"

    def test_critical_prediction(self):
        """Extreme values should predict CRITICAL."""
        import joblib
        model_data = joblib.load(CLASSIFIER_PATH)
        scaler = joblib.load(SCALER_PATH)
        clf = model_data["model"]
        le = model_data["label_encoder"]

        # Critical: extreme speed, high impact
        X = np.array([[180, 10.0, 0.9, 5, 0]])
        X_scaled = scaler.transform(X)
        pred = le.inverse_transform(clf.predict(X_scaled))[0]
        assert pred == STATUS_CRITICAL, f"Expected CRITICAL, got {pred}"


class TestExplanations:
    """Tests for the explanation engine."""

    def test_normal_explanation(self):
        row = pd.Series({
            "speed": 50, "acceleration": 0.5, "impact_score": 0.02,
            "battery_status": 90, "gps_status": 1,
        })
        explanations = generate_explanation(row)
        assert len(explanations) == 1
        assert "normal" in explanations[0].lower()

    def test_speed_warning_explanation(self):
        row = pd.Series({
            "speed": 120, "acceleration": 0.5, "impact_score": 0.02,
            "battery_status": 90, "gps_status": 1,
        })
        explanations = generate_explanation(row)
        assert any("speed" in e.lower() for e in explanations)

    def test_gps_loss_explanation(self):
        row = pd.Series({
            "speed": 50, "acceleration": 0.5, "impact_score": 0.02,
            "battery_status": 90, "gps_status": 0,
        })
        explanations = generate_explanation(row)
        assert any("gps" in e.lower() for e in explanations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
