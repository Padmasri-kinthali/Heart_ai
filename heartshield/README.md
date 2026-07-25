# HeartShield — Early Heart Disease Risk Prediction

HeartShield estimates cardiovascular risk from clinical values *before* severe
symptoms appear. It accepts manual health input, extracts values from
uploaded lab reports (PDF/image) via OCR, engineers clinically-motivated
features, runs a trained ML model, and returns a **Low / Moderate / High**
risk category with tailored preventive recommendations.

> ⚠️ **Not a medical device.** This is a decision-support screening tool
> trained on a public research dataset. It does not diagnose disease and
> must not replace consultation with a qualified clinician.

## How it works

```
                ┌─────────────────────┐
   Manual entry │                     │
   ────────────►│                     │
                │   Streamlit UI      │
   PDF / Image  │   (app.py)          │
   ────────────►│                     │
                └─────────┬───────────┘
                          │
                 OCR extraction
           (heartshield/ocr_extractor.py)
        pypdf (digital PDFs) + pdf2image
        + pytesseract (scanned PDFs / images)
        + regex parsing of lab values
                          │
                          ▼
              Clean & prefill form fields
             (user reviews/edits everything)
                          │
                          ▼
             Feature engineering
      (heartshield/feature_engineering.py)
   hr_reserve, hr_pct_of_max, pulse_pressure_proxy,
   chol/bp/age risk flags, ST-depression severity bucket
                          │
                          ▼
          ML Pipeline (model/heart_model.joblib)
    Impute → Scale/OneHot → Calibrated classifier
   (Logistic Regression / Random Forest / Gradient
    Boosting — best picked by 5-fold CV ROC-AUC)
                          │
                          ▼
         Risk engine (heartshield/risk_engine.py)
   probability → Low/Moderate/High + lifestyle nudge
    (smoking, family history, diabetes, BMI, activity)
                          │
                          ▼
        Risk label + preventive recommendations
```

## Project structure

```
heartshield/
├── app.py                        # Streamlit application (entry point)
├── train_model.py                # Trains & saves the ML model
├── requirements.txt
├── data/
│   └── heart.csv                 # UCI Cleveland Heart Disease dataset (303 patients)
├── model/
│   ├── heart_model.joblib        # Trained pipeline (created by train_model.py)
│   ├── metrics.json              # Evaluation metrics
│   └── feature_importance.png
└── heartshield/                  # Shared package (imported by training + app)
    ├── feature_engineering.py    # HeartFeatureEngineer transformer
    ├── ocr_extractor.py          # PDF/image → text → parsed clinical fields
    └── risk_engine.py            # Risk categorization + recommendations
```

## Setup

```bash
pip install -r requirements.txt

# System dependency for OCR (Ubuntu/Debian):
sudo apt-get install tesseract-ocr poppler-utils
# macOS:
brew install tesseract poppler
```

## Train the model

```bash
python train_model.py
```

This trains Logistic Regression, Random Forest, and Gradient Boosting on the
13 clinical features (age, sex, chest pain type, resting BP, cholesterol,
fasting blood sugar, resting ECG, max heart rate, exercise angina, ST
depression, ST slope, vessels colored, thalassemia result), 5-fold
cross-validates on ROC-AUC, picks the best, calibrates its probabilities
(`CalibratedClassifierCV`), and saves the full pipeline to
`model/heart_model.joblib`.

Current held-out test performance (your numbers may vary slightly on retrain):

| Metric | Score |
|---|---|
| Accuracy | ~0.89 |
| Precision | ~0.84 |
| Recall | ~0.93 |
| F1 | ~0.88 |
| ROC-AUC | ~0.96 |

## Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Design notes & honesty about limitations

- **OCR extraction is intentionally conservative.** Routine lab reports
  typically only contain age, blood pressure, cholesterol, fasting blood
  sugar, and sometimes heart rate. Exam-derived fields (chest pain type,
  exercise-induced angina, ST depression, vessel count, thalassemia result)
  come from a clinical/stress-test workup and are **not** reliably printed
  on patient-facing reports, so OCR never fabricates them — the user always
  fills those in manually, and every OCR-prefilled value is shown for
  review before prediction.
- **Lifestyle factors** (smoking, family history, diabetes, BMI, activity
  level) aren't in the training dataset, so instead of feeding them into the
  ML model directly (which would silently change its calibration), they
  apply a small, capped, transparent adjustment to the predicted
  probability and directly generate matching recommendations.
- **Dataset**: UCI Machine Learning Repository, "Heart Disease" dataset
  (Cleveland processed subset, 303 instances), Janosi, Steinbrunn, Pfisterer
  & Detrano, 1989. Licensed CC BY 4.0.
- **Risk thresholds** (Low < 30%, Moderate 30–60%, High ≥ 60%) are a
  reasonable operating point for a screening tool but are not derived from
  a formal clinical validation study — treat them as configurable, not
  gospel.

## Extending it

- Swap in a larger/more diverse dataset (e.g. merge Hungarian, Switzerland,
  Long Beach VA cohorts) for better generalization beyond one hospital.
- Add authentication + persistent user history if deploying beyond a demo.
- Replace the regex-based OCR parser with a layout-aware model (e.g.
  table-detection) for structured lab report PDFs.
- Add SHAP-based per-prediction explanations for clinician-facing transparency.
