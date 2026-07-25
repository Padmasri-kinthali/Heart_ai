"""
HeartShield - Model Training
-----------------------------
Trains and compares several classifiers for early heart-disease risk
prediction on the UCI Cleveland Heart Disease dataset, picks the best
performer via cross-validated ROC-AUC, calibrates its probabilities,
and saves the full pipeline (feature engineering + preprocessing +
model) as a single artifact so inference code never has to re-derive
feature logic.

Run:
    python train_model.py
Outputs:
    model/heart_model.joblib   - full sklearn Pipeline (predict_proba ready)
    model/metrics.json         - evaluation metrics for the report/README
    model/feature_importance.png
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartshield.feature_engineering import (
    FINAL_CATEGORICAL,
    FINAL_FLAGS,
    FINAL_NUMERIC,
    HeartFeatureEngineer,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    flag_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, FINAL_NUMERIC),
        ("cat", categorical_pipe, FINAL_CATEGORICAL),
        ("flag", flag_pipe, FINAL_FLAGS),
    ])


def build_candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE,
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    fitted_pipelines = {}

    for name, clf in build_candidates().items():
        pipe = Pipeline([
            ("engineer", HeartFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("classifier", clf),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        results[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }
        print(f"{name:20s} CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    print(f"\nBest model: {best_name}")

    # Calibrate probabilities of the best model for more trustworthy risk scores
    best_raw_clf = build_candidates()[best_name]
    calibrated_pipe = Pipeline([
        ("engineer", HeartFeatureEngineer()),
        ("preprocess", build_preprocessor()),
    ])
    X_train_transformed = calibrated_pipe.fit_transform(X_train, y_train)
    calibrated_clf = CalibratedClassifierCV(best_raw_clf, method="sigmoid", cv=5)
    calibrated_clf.fit(X_train_transformed, y_train)

    final_pipeline = Pipeline([
        ("engineer", HeartFeatureEngineer()),
        ("preprocess", calibrated_pipe.named_steps["preprocess"]),
        ("classifier", calibrated_clf),
    ])

    # Evaluate on held-out test set
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    print("\nHeld-out test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:10s}: {v:.3f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Save everything
    joblib.dump(final_pipeline, MODEL_DIR / "heart_model.joblib")

    metrics_out = {
        "model_selected": best_name,
        "cv_results": results,
        "test_metrics": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Feature importance plot (best-effort; only for tree models)
    try:
        raw_clf_refit = build_candidates()[best_name]
        raw_clf_refit.fit(X_train_transformed, y_train)
        if hasattr(raw_clf_refit, "feature_importances_"):
            preprocessor = calibrated_pipe.named_steps["preprocess"]
            feature_names = preprocessor.get_feature_names_out()
            importances = raw_clf_refit.feature_importances_
            order = np.argsort(importances)[-15:]
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(order)), importances[order])
            plt.yticks(range(len(order)), [feature_names[i] for i in order])
            plt.xlabel("Importance")
            plt.title(f"Top Feature Importances ({best_name})")
            plt.tight_layout()
            plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
            print(f"\nSaved feature importance plot -> {MODEL_DIR / 'feature_importance.png'}")
    except Exception as e:
        print(f"(skipped feature importance plot: {e})")

    print(f"\nSaved trained pipeline -> {MODEL_DIR / 'heart_model.joblib'}")
    print(f"Saved metrics -> {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
