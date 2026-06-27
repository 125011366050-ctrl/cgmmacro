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


st.set_page_config(page_title="CDSS - Glucose Predictor", layout="centered")
st.title("🩺 Clinical Decision Support System")
st.subheader("Glucose Risk Prediction")

try:
    system = load_system()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.markdown("---")
current_glucose = st.number_input(
    "Enter Current Blood Glucose (mg/dL)",
    min_value=50,
    max_value=400,
    value=120,
    step=1
)

if st.button("Run Prediction"):
    try:
        X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
        sample = X_test[0:1]

        result = system.run(current_glucose, sample)
        preds = result["predictions"]
        risk = result["risk"]

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("30-min Glucose", f"{preds[0]:.1f} mg/dL")
        col2.metric("60-min Glucose", f"{preds[1]:.1f} mg/dL")
        col3.metric("90-min Glucose", f"{preds[2]:.1f} mg/dL")

        st.markdown("---")
        st.subheader("⚠️ Risk Assessment")

        risk_level = risk["risk"]
        spike = risk["spike"]
        peak = risk["peak"]

        if risk_level == "HIGH":
            st.error(f"🔴 Risk Level: HIGH | Peak: {peak:.1f} mg/dL | Spike: +{spike:.1f}")
        elif risk_level == "MEDIUM":
            st.warning(f"🟡 Risk Level: MEDIUM | Peak: {peak:.1f} mg/dL | Spike: +{spike:.1f}")
        else:
            st.success(f"🟢 Risk Level: LOW | Peak: {peak:.1f} mg/dL | Spike: +{spike:.1f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
