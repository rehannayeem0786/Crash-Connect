"""
Multi-model comparison for the CATS AI Safety Bot.

Trains and compares: Random Forest, Gradient Boosting, and SVM
to demonstrate thorough ML methodology.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import joblib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import FEATURE_COLUMNS, TARGET_COLUMN, CLASSIFIER_PATH, SCALER_PATH, DATASET_PATH


def train_and_compare():
    """Train multiple models and compare performance."""
    print("=" * 70)
    print("CATS AI Safety Bot — Multi-Model Comparison")
    print("=" * 70)

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

    print(f"\nDataset: {len(df)} samples ({X_train.shape[0]} train / {X_test.shape[0]} test)")
    print(f"Classes: {le.classes_.tolist()}\n")

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True, random_state=42
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"Training: {name}")
        print(f"{'─' * 50}")

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring="f1_weighted")

        results[name] = {
            "accuracy": float(accuracy),
            "f1_weighted": float(f1),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "train_time_s": float(train_time),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"  Precision:    {precision:.4f}")
        print(f"  Recall:       {recall:.4f}")
        print(f"  CV F1:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Train time:   {train_time:.2f}s")
        print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # Find best model
    best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
    print(f"\n{'=' * 70}")
    print(f"🏆 BEST MODEL: {best_name} (CV F1 = {results[best_name]['cv_f1_mean']:.4f})")
    print(f"{'=' * 70}")

    # Save comparison results
    comparison = {
        "models": results,
        "best_model": best_name,
        "class_labels": le.classes_.tolist(),
    }
    comp_path = os.path.join(os.path.dirname(__file__), "model_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {comp_path}")

    # Retrain best model on full training set and save
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    joblib.dump({"model": best_model, "label_encoder": le}, CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Best model saved to {CLASSIFIER_PATH}")

    # Also save Gradient Boosting for ensemble/comparison display
    gb_model = models["Gradient Boosting"]
    gb_path = os.path.join(os.path.dirname(__file__), "gradient_boosting.pkl")
    joblib.dump({"model": gb_model, "label_encoder": le}, gb_path)

    return comparison


if __name__ == "__main__":
    train_and_compare()
