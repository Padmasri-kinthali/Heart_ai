"""
HeartShield - Early Heart Disease Risk Prediction
----------------------------------------------------
Streamlit application tying together:
  1. Manual health data entry
  2. PDF/image medical report upload -> OCR extraction (prefill, reviewable)
  3. Feature engineering (heartshield/feature_engineering.py)
  4. ML risk prediction (model/heart_model.joblib, trained by train_model.py)
  5. Low / Moderate / High risk display
  6. Preventive healthcare recommendations (heartshield/risk_engine.py)

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from heartshield.ocr_extractor import extract_from_file
from heartshield.risk_engine import adjust_probability, get_recommendations

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "heart_model.joblib"

st.set_page_config(page_title="HeartShield - Heart Disease Risk", page_icon="🫀", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def init_state():
    defaults = dict(
        age=50, sex="Male", cp="Typical angina", trestbps=120, chol=200,
        fbs="No (<=120 mg/dl)", restecg="Normal", thalach=150, exang="No",
        oldpeak=0.0, slope="Upsloping", ca=0, thal="Normal",
        smoker=False, family_history=False, diabetic=False, sedentary=False, bmi=24.0,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

CP_MAP = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
RESTECG_MAP = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE_MAP = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_MAP = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}

st.title("🫀 HeartShield")
st.caption("Early heart-disease risk screening -- decision support, not a diagnosis.")

st.info(
    "HeartShield estimates cardiovascular risk from clinical values using a model trained on "
    "historical patient data. It is **not a medical diagnosis**. Always discuss results with a "
    "qualified clinician, especially if your risk comes back Moderate or High.",
    icon="ℹ️",
)

tab_upload, tab_manual, tab_result = st.tabs(["📄 Upload Report", "✍️ Manual Entry", "📊 Risk Result"])

# -----------------------------------------------------------------
# TAB 1: Upload PDF/Image -> OCR -> prefill session state
# -----------------------------------------------------------------
with tab_upload:
    st.subheader("Upload a medical report")
    st.write(
        "Upload a lab report or vitals printout (PDF, JPG, or PNG). HeartShield will try to "
        "auto-extract values it recognizes. **You must review and confirm everything in the "
        "Manual Entry tab before predicting** -- OCR is a convenience, not a guarantee."
    )
    uploaded = st.file_uploader("Medical report", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Extracting text and parsing clinical values..."):
            try:
                result = extract_from_file(tmp_path)
            except Exception as e:
                result = None
                st.error(f"Could not process this file: {e}")

        if result is not None:
            for note in result.confidence_notes:
                st.write("- " + note)

            if result.fields:
                st.success("Extracted values have been pre-filled in the Manual Entry tab. Please verify them.")
                f = result.fields
                if "age" in f:
                    st.session_state["age"] = int(f["age"])
                if "sex" in f and f["sex"] is not None:
                    st.session_state["sex"] = "Male" if f["sex"] == 1 else "Female"
                if "trestbps" in f:
                    st.session_state["trestbps"] = int(f["trestbps"])
                if "chol" in f:
                    st.session_state["chol"] = int(f["chol"])
                if "fbs" in f:
                    st.session_state["fbs"] = "Yes (>120 mg/dl)" if f["fbs"] == 1 else "No (<=120 mg/dl)"
                if "thalach" in f:
                    st.session_state["thalach"] = int(f["thalach"])

            with st.expander("Show raw extracted text"):
                st.text(result.raw_text or "(no text extracted)")

# -----------------------------------------------------------------
# TAB 2: Manual entry (also shows OCR-prefilled values for review)
# -----------------------------------------------------------------
with tab_manual:
    st.subheader("Health information")
    st.caption("Fields prefilled from an uploaded report are editable -- please double-check them.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, key="age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
        cp = st.selectbox("Chest pain type", list(CP_MAP.keys()), key="cp")
        trestbps = st.number_input("Resting blood pressure (mmHg, systolic)", min_value=60, max_value=250, key="trestbps")
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=700, key="chol")
        fbs = st.selectbox("Fasting blood sugar", ["No (<=120 mg/dl)", "Yes (>120 mg/dl)"], key="fbs")
        restecg = st.selectbox("Resting ECG result", list(RESTECG_MAP.keys()), key="restecg")
    with col2:
        thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=250, key="thalach")
        exang = st.selectbox("Exercise-induced angina", ["No", "Yes"], key="exang")
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak")
        slope = st.selectbox("Slope of peak exercise ST segment", list(SLOPE_MAP.keys()), key="slope")
        ca = st.selectbox("Major vessels colored by fluoroscopy (0-3)", [0, 1, 2, 3], key="ca")
        thal = st.selectbox("Thalassemia result", list(THAL_MAP.keys()), key="thal")

    st.markdown("---")
    st.subheader("Lifestyle & history (not in the clinical model, used to fine-tune your result)")
    lcol1, lcol2 = st.columns(2)
    with lcol1:
        smoker = st.checkbox("Current smoker", key="smoker")
        family_history = st.checkbox("Family history of heart disease", key="family_history")
        diabetic = st.checkbox("Diagnosed diabetic", key="diabetic")
    with lcol2:
        sedentary = st.checkbox("Mostly sedentary (little regular exercise)", key="sedentary")
        bmi = st.number_input("BMI (optional)", min_value=10.0, max_value=60.0, step=0.1, key="bmi")

    predict_clicked = st.button("🔍 Predict My Heart Disease Risk", type="primary", use_container_width=True)

# -----------------------------------------------------------------
# TAB 3: Prediction + recommendations
# -----------------------------------------------------------------
with tab_result:
    model = load_model()

    if model is None:
        st.warning("Model not found. Run `python train_model.py` first to train and save the model.")
    elif not predict_clicked and "last_result" not in st.session_state:
        st.write("Fill in the Manual Entry tab and click **Predict My Heart Disease Risk** to see results here.")
    else:
        if predict_clicked:
            input_row = pd.DataFrame([{
                "age": st.session_state["age"],
                "sex": 1 if st.session_state["sex"] == "Male" else 0,
                "cp": CP_MAP[st.session_state["cp"]],
                "trestbps": st.session_state["trestbps"],
                "chol": st.session_state["chol"],
                "fbs": 1 if st.session_state["fbs"].startswith("Yes") else 0,
                "restecg": RESTECG_MAP[st.session_state["restecg"]],
                "thalach": st.session_state["thalach"],
                "exang": 1 if st.session_state["exang"] == "Yes" else 0,
                "oldpeak": st.session_state["oldpeak"],
                "slope": SLOPE_MAP[st.session_state["slope"]],
                "ca": st.session_state["ca"],
                "thal": THAL_MAP[st.session_state["thal"]],
            }])

            base_proba = float(model.predict_proba(input_row)[0, 1])

            engineer = model.named_steps["engineer"]
            engineered = engineer.transform(input_row).iloc[0].to_dict()
            engineered_flags = {
                "chol_risk_flag": bool(engineered.get("chol_risk_flag")),
                "bp_risk_flag": bool(engineered.get("bp_risk_flag")),
                "age_risk_flag": bool(engineered.get("age_risk_flag")),
                "vessel_risk": bool(engineered.get("vessel_risk")),
            }

            lifestyle = {
                "smoker": st.session_state["smoker"],
                "family_history": st.session_state["family_history"],
                "diabetic": st.session_state["diabetic"],
                "sedentary": st.session_state["sedentary"],
                "high_bmi": st.session_state["bmi"] >= 30,
            }

            risk_result = adjust_probability(base_proba, lifestyle)
            active_lifestyle = [k for k, v in lifestyle.items() if v]
            recommendations = get_recommendations(risk_result.risk_label, engineered_flags, active_lifestyle)

            st.session_state["last_result"] = dict(
                risk_result=risk_result,
                recommendations=recommendations,
                base_proba=base_proba,
            )

        data = st.session_state["last_result"]
        risk_result = data["risk_result"]
        label = risk_result.risk_label

        color = {"Low Risk": "green", "Moderate Risk": "orange", "High Risk": "red"}[label]
        st.markdown(f"## Risk Level: :{color}[{label}]")
        st.progress(min(risk_result.adjusted_probability, 1.0))
        st.write(f"Estimated risk score: **{risk_result.adjusted_probability * 100:.1f}%** "
                 f"(clinical model: {risk_result.base_probability * 100:.1f}%"
                 + (f", adjusted for: {', '.join(risk_result.contributing_lifestyle_factors)}"
                    if risk_result.contributing_lifestyle_factors else "") + ")")

        st.markdown("### Preventive recommendations")
        for rec in data["recommendations"]:
            st.markdown(f"- {rec}")

        st.markdown("---")
        st.caption(
            "HeartShield is a screening aid trained on historical clinical data (UCI Cleveland "
            "Heart Disease dataset) and does not account for every individual factor. It cannot "
            "replace an in-person clinical evaluation."
        )
