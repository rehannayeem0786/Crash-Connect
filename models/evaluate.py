"""
Model evaluation script — generates detailed metrics and plots.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import FEATURE_COLUMNS, TARGET_COLUMN, CLASSIFIER_PATH, SCALER_PATH, DATASET_PATH


def evaluate():
    """Load trained model and generate evaluation plots."""
    # Load data
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Load model
    model_data = joblib.load(CLASSIFIER_PATH)
    clf = model_data["model"]

    y_pred = clf.predict(X_test)

    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — CATS Risk Classifier", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    # Feature Importance Plot
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_COLUMNS[i] for i in sorted_idx],
        importances[sorted_idx],
        color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importances", fontsize=14)
    plt.tight_layout()
    fi_path = os.path.join(os.path.dirname(__file__), "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Feature importance saved: {fi_path}")

    print("\n✅ Evaluation plots generated!")


if __name__ == "__main__":
    evaluate()
