"""
HeartShield - Risk Categorization & Recommendation Engine
------------------------------------------------------------
Turns the ML model's calibrated probability into a Low / Moderate / High
risk label, applies a small transparent adjustment for lifestyle factors
the trained model never saw (BMI, smoking, family history, activity
level, diabetes history), and produces tailored preventive-care
recommendations.

This is decision support, not a diagnosis. Every output must be shown
to the user alongside a clear disclaimer to consult a clinician.
"""

from dataclasses import dataclass, field
from typing import Optional


LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60

# Small, capped, additive adjustments (probability points) for factors
# outside the trained model. Kept intentionally modest -- lifestyle
# context should nudge the score, not override the clinical model.
LIFESTYLE_WEIGHTS = {
    "smoker": 0.06,
    "family_history": 0.05,
    "diabetic": 0.06,
    "sedentary": 0.04,
    "high_bmi": 0.05,  # BMI >= 30
}
MAX_LIFESTYLE_ADJUSTMENT = 0.20


@dataclass
class RiskResult:
    base_probability: float
    adjusted_probability: float
    risk_label: str
    contributing_lifestyle_factors: list = field(default_factory=list)


def adjust_probability(base_probability: float, lifestyle: dict) -> RiskResult:
    """lifestyle: dict of booleans, e.g. {'smoker': True, 'family_history': False, ...}"""
    adjustment = 0.0
    contributing = []
    for factor, weight in LIFESTYLE_WEIGHTS.items():
        if lifestyle.get(factor):
            adjustment += weight
            contributing.append(factor)
    adjustment = min(adjustment, MAX_LIFESTYLE_ADJUSTMENT)

    adjusted = min(base_probability + adjustment, 0.99)
    label = categorize(adjusted)
    return RiskResult(
        base_probability=base_probability,
        adjusted_probability=adjusted,
        risk_label=label,
        contributing_lifestyle_factors=contributing,
    )


def categorize(probability: float) -> str:
    if probability < LOW_THRESHOLD:
        return "Low Risk"
    elif probability < HIGH_THRESHOLD:
        return "Moderate Risk"
    return "High Risk"


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

GENERAL_RECOMMENDATIONS = [
    "Schedule a routine check-up with a primary care physician to discuss these results.",
    "Track blood pressure and cholesterol periodically rather than as one-off readings.",
    "Aim for at least 150 minutes of moderate aerobic activity per week, as tolerated.",
    "Favor a diet rich in vegetables, whole grains, and lean protein; limit sodium and saturated fat.",
]

MODERATE_RECOMMENDATIONS = [
    "Discuss a formal cardiovascular risk assessment (e.g. lipid panel, ECG) with a doctor.",
    "Monitor blood pressure at home weekly and keep a log to share at appointments.",
    "If you smoke, ask a clinician about smoking-cessation support.",
    "Review fasting blood sugar; consider screening for prediabetes if not already done.",
]

HIGH_RECOMMENDATIONS = [
    "Seek a clinical evaluation from a cardiologist promptly -- this tool flags elevated risk, it does not diagnose.",
    "Ask about a stress test, echocardiogram, or coronary calcium scoring if not already performed.",
    "Discuss medication options (e.g. statins, antihypertensives) if indicated by your clinician.",
    "Avoid strenuous, unsupervised exertion until a clinician has reviewed your cardiac status.",
]

FACTOR_SPECIFIC = {
    "chol_risk_flag": "Your cholesterol reading is in the high range (>=240 mg/dl) -- ask about dietary changes or lipid-lowering therapy.",
    "bp_risk_flag": "Your resting blood pressure reading is elevated (>=140 mmHg systolic) -- home monitoring and follow-up are advised.",
    "age_risk_flag": "Cardiovascular risk naturally rises with age -- routine annual screening is worthwhile from here on.",
    "vessel_risk": "Imaging indicated involvement of one or more major vessels -- this warrants specialist follow-up.",
    "smoker": "Smoking is one of the largest modifiable heart-disease risk factors -- cessation has an outsized benefit.",
    "diabetic": "Diabetes significantly compounds cardiovascular risk -- tight glucose control matters for heart health too.",
    "high_bmi": "A BMI in the obese range (30+) is linked to higher cardiovascular risk -- gradual, sustainable weight loss helps.",
    "sedentary": "Low physical activity is a modifiable risk factor -- even brisk walking most days makes a measurable difference.",
    "family_history": "A family history of heart disease raises baseline risk -- mention this explicitly to your doctor.",
}


def get_recommendations(risk_label: str, engineered_flags: dict, lifestyle_factors: list) -> list:
    if risk_label == "Low Risk":
        recs = list(GENERAL_RECOMMENDATIONS)
    elif risk_label == "Moderate Risk":
        recs = list(GENERAL_RECOMMENDATIONS) + list(MODERATE_RECOMMENDATIONS)
    else:
        recs = list(GENERAL_RECOMMENDATIONS) + list(HIGH_RECOMMENDATIONS)

    for flag_name, is_active in engineered_flags.items():
        if is_active and flag_name in FACTOR_SPECIFIC:
            recs.append(FACTOR_SPECIFIC[flag_name])

    for factor in lifestyle_factors:
        if factor in FACTOR_SPECIFIC:
            recs.append(FACTOR_SPECIFIC[factor])

    # de-duplicate, preserve order
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)
    return unique_recs
