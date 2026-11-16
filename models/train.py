"""
Model training script for the CATS AI Safety Bot risk classifier.

Trains a Random Forest classifier on simulated vehicle telemetry data
to classify readings as NORMAL, WARNING, or CRITICAL.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import FEATURE_COLUMNS, TARGET_COLUMN, CLASSIFIER_PATH, SCALER_PATH, DATASET_PATH


def load_data():
    """Load and prepare the telemetry dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Label distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}\n")
    return df


def train_model(df):
    """Train the Random Forest classifier."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples\n")

    # Train Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # === Evaluation ===
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print(f"\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    # Cross-validation
    print(f"\n5-Fold Cross-Validation:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=cv, scoring="f1_weighted")
    print(f"  F1 scores: {cv_scores.round(4)}")
    print(f"  Mean F1:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    print(f"\nFeature Importances:")
    importances = clf.feature_importances_
    feature_importance = sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
    )
    for feat, imp in feature_importance:
        bar = "█" * int(imp * 50)
        print(f"  {feat:20s} {imp:.4f} {bar}")

    # Save evaluation metrics
    metrics = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "feature_importances": {feat: float(imp) for feat, imp in feature_importance},
        "class_labels": le.classes_.tolist(),
    }
    metrics_path = os.path.join(os.path.dirname(__file__), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return clf, scaler, le


def save_model(clf, scaler, le):
    """Save the trained model and scaler."""
    joblib.dump({"model": clf, "label_encoder": le}, CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {CLASSIFIER_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


def main():
    df = load_data()
    clf, scaler, le = train_model(df)
    save_model(clf, scaler, le)
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
