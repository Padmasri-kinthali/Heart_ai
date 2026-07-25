"""
HeartShield - Feature Engineering
----------------------------------
Shared transformer used by BOTH training (model/train_model.py) and
inference (app/app.py) so the exact same feature-construction logic is
applied every time. This guarantees no train/serve skew.

Base clinical features (from the UCI Cleveland dataset):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal

Engineered features added on top:
    hr_reserve       - predicted max HR (220-age) minus achieved max HR
    hr_pct_of_max     - thalach as a % of age-predicted max HR
    pulse_pressure_proxy - trestbps relative to a normal-BP baseline (120)
    chol_risk_flag    - 1 if cholesterol >= 240 mg/dl (high, per ACC/AHA)
    bp_risk_flag      - 1 if resting BP >= 140 mmHg (stage-2 hypertension range)
    age_risk_flag     - 1 if age >= 55
    st_severity       - bucketed oldpeak (ST depression) severity 0-3
    vessel_risk       - 1 if ca (major vessels colored) >= 1
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BASE_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ENGINEERED_NUMERIC_FEATURES = ["hr_reserve", "hr_pct_of_max", "pulse_pressure_proxy"]
ENGINEERED_FLAG_FEATURES = ["chol_risk_flag", "bp_risk_flag", "age_risk_flag", "st_severity", "vessel_risk"]

ALL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ["st_severity"]  # st_severity treated as categorical bucket
FINAL_NUMERIC = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
FINAL_FLAGS = ["chol_risk_flag", "bp_risk_flag", "age_risk_flag", "vessel_risk"]
FINAL_CATEGORICAL = BASE_CATEGORICAL_FEATURES + ["st_severity"]

RAW_REQUIRED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def _st_severity_bucket(oldpeak: float) -> int:
    if oldpeak < 1:
        return 0  # none / minimal
    elif oldpeak < 2:
        return 1  # mild
    elif oldpeak < 4:
        return 2  # moderate
    return 3      # severe


class HeartFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer: raw clinical fields -> engineered feature frame."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in RAW_REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["trestbps"] = pd.to_numeric(df["trestbps"], errors="coerce")
        df["chol"] = pd.to_numeric(df["chol"], errors="coerce")
        df["thalach"] = pd.to_numeric(df["thalach"], errors="coerce")
        df["oldpeak"] = pd.to_numeric(df["oldpeak"], errors="coerce")

        predicted_max_hr = 220 - df["age"]
        df["hr_reserve"] = predicted_max_hr - df["thalach"]
        df["hr_pct_of_max"] = (df["thalach"] / predicted_max_hr.replace(0, np.nan)) * 100
        df["pulse_pressure_proxy"] = df["trestbps"] - 120

        df["chol_risk_flag"] = (df["chol"] >= 240).astype(int)
        df["bp_risk_flag"] = (df["trestbps"] >= 140).astype(int)
        df["age_risk_flag"] = (df["age"] >= 55).astype(int)
        df["st_severity"] = df["oldpeak"].apply(_st_severity_bucket)
        df["vessel_risk"] = (pd.to_numeric(df["ca"], errors="coerce") >= 1).astype(int)

        keep_cols = FINAL_NUMERIC + FINAL_FLAGS + FINAL_CATEGORICAL
        # avoid duplicate 'st_severity' (present in both flags calc and categorical)
        keep_cols = list(dict.fromkeys(keep_cols))
        return df[keep_cols]
