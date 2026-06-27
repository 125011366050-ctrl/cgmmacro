import streamlit as st
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from reco import ClinicalOrchestrator, Config


@st.cache_resource
def load_system():
    config = Config()
    config.DATA_PATH = BASE_DIR
    config.INPUT_SIZE = 18
    return ClinicalOrchestrator(config)


st.set_page_config(page_title="CDSS - Glucose Predictor", layout="wide")
st.title("🩺 Clinical Decision Support System")
st.subheader("CGM-Based Glucose Risk Prediction")

try:
    system = load_system()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.markdown("---")
st.subheader("📋 Patient Input")

col_a, col_b = st.columns(2)

with col_a:
    current_glucose = st.number_input(
        "Current Blood Glucose (mg/dL)",
        min_value=50,
        max_value=400,
        value=120,
        step=1
    )
    meal_carbs = st.number_input(
        "Recent Meal Carbs (grams)",
        min_value=0,
        max_value=300,
        value=60,
        step=5
    )
    insulin_dose = st.number_input(
        "Recent Insulin Dose (units)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5
    )

with col_b:
    st.markdown("**Last 10 CGM Readings (mg/dL) — oldest to newest**")
    cgm_inputs = []
    cgm_cols = st.columns(5)
    defaults = [105, 108, 112, 115, 118, 119, 120, 120, 120, 120]
    for i in range(10):
        with cgm_cols[i % 5]:
            val = st.number_input(
                f"t-{10 - i}",
                min_value=50,
                max_value=400,
                value=defaults[i],
                step=1,
                key=f"cgm_{i}"
            )
            cgm_inputs.append(val)

st.markdown("---")

if st.button("🔍 Run Prediction", use_container_width=True):
    try:
        # Build input sequence (1, 30, 18)
        # Fill window with CGM trend + clinical features
        window_size = 30
        n_features = 18

        sample = np.zeros((1, window_size, n_features), dtype=np.float32)

        # Feature 0: glucose trend (repeat last 10 CGM across 30 steps)
        cgm_array = np.array(cgm_inputs, dtype=np.float32)
        trend = np.interp(
            np.linspace(0, 9, window_size),
            np.arange(10),
            cgm_array
        )
        sample[0, :, 0] = trend

        # Feature 1: current glucose repeated
        sample[0, :, 1] = current_glucose

        # Feature 2: carbs
        sample[0, :, 2] = meal_carbs

        # Feature 3: insulin
        sample[0, :, 3] = insulin_dose

        # Feature 4: glucose delta (rate of change)
        delta = current_glucose - cgm_inputs[-2] if len(cgm_inputs) >= 2 else 0
        sample[0, :, 4] = delta

        # Features 5-17: zeros (other CGM/clinical signals not collected here)

        result = system.run(current_glucose, sample)
        preds = result["predictions"]
        risk = result["risk"]

        st.subheader("📊 Predicted Glucose Levels")
        col1, col2, col3 = st.columns(3)
        col1.metric("30 min", f"{preds[0]:.1f} mg/dL",
                    delta=f"{preds[0]-current_glucose:+.1f}")
        col2.metric("60 min", f"{preds[1]:.1f} mg/dL",
                    delta=f"{preds[1]-current_glucose:+.1f}")
        col3.metric("90 min", f"{preds[2]:.1f} mg/dL",
                    delta=f"{preds[2]-current_glucose:+.1f}")

        st.markdown("---")
        st.subheader("⚠️ Risk Assessment")

        risk_level = risk["risk"]
        spike = risk["spike"]
        peak = risk["peak"]

        if risk_level == "HIGH":
            st.error(f"🔴 HIGH RISK — Peak: {peak:.1f} mg/dL | Spike: +{spike:.1f} mg/dL")
            st.warning("⚡ Consider corrective insulin or immediate clinical review.")
        elif risk_level == "MEDIUM":
            st.warning(f"🟡 MEDIUM RISK — Peak: {peak:.1f} mg/dL | Spike: +{spike:.1f} mg/dL")
            st.info("📋 Monitor closely. Avoid additional carbohydrates.")
        else:
            st.success(f"🟢 LOW RISK — Peak: {peak:.1f} mg/dL | Spike: +{spike:.1f} mg/dL")
            st.info("✅ Glucose trajectory appears stable.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
