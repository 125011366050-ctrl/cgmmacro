import streamlit as st
import numpy as np
import torch
import joblib
from datetime import datetime
from model import LSTMWithPredictionHead
from pytorch_tabnet.tab_model import TabNetRegressor

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="CGM AI System", page_icon="🩸")
st.title("🩸 CGM Prediction System (LSTM + TabNet)")
st.write("Predict glucose levels for next 30, 60, 120 minutes")

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    cfg = joblib.load("cgmacros_cleaned/model_config.pkl")

    lstm = LSTMWithPredictionHead(
        cfg["input_size"],
        cfg["hidden_size"],
        cfg["n_horizons"]
    )

    lstm.load_state_dict(
        torch.load("cgmacros_cleaned/lstm_model.pth", map_location="cpu")
    )
    lstm.eval()

    tabnet = TabNetRegressor()
    tabnet.load_model("cgmacros_cleaned/tabnet_on_learned_embeddings.zip")

    feature_scaler = joblib.load("cgmacros_cleaned/feature_scaler.pkl")
    glucose_scaler = joblib.load("cgmacros_cleaned/glucose_scaler.pkl")

    return cfg, lstm, tabnet, feature_scaler, glucose_scaler

cfg, lstm, tabnet, feature_scaler, glucose_scaler = load_models()

WINDOW = cfg["window_size"] if "window_size" in cfg else 36

# ----------------------------
# INPUT UI
# ----------------------------
st.header("Patient Inputs")

glucose = st.number_input("Current Glucose (mg/dL)", 40.0, 400.0, 110.0)
hr = st.number_input("Heart Rate (bpm)", 40.0, 180.0, 80.0)
meal = st.radio("Meal Taken?", ["Yes", "No"])
carbs = st.number_input("Carbohydrates (g)", 0.0, 300.0, 40.0)
protein = st.number_input("Protein (g)", 0.0, 200.0, 20.0)
fat = st.number_input("Fat (g)", 0.0, 200.0, 10.0)

# Time features
now = datetime.now()
hour = now.hour
day = now.weekday()
isnight = 1 if (hour < 6 or hour >= 22) else 0

meal_flag = 1 if meal == "Yes" else 0

# ----------------------------
# FEATURE VECTOR (MUST MATCH TRAINING ORDER)
# ----------------------------
features = np.array([[
    glucose,          # GL
    glucose,          # GL_MA_5
    5.0,              # GL_STD_5
    0.0,              # GL_Diff
    0.0,              # GL_Slope
    0.0,              # GL_Acceleration
    0.1,              # CV_Glucose
    hr,               # HR
    hr,               # HR_MA_5
    meal_flag,       # Meal_Flag
    carbs,           # Calories (approx)
    carbs,           # Carbs
    protein,
    fat,
    5.0,              # Fiber placeholder
    hour,
    day,
    isnight
]], dtype=np.float32)

# ----------------------------
# PREDICT
# ----------------------------
if st.button("Predict Glucose"):

    # Scale features
    row_scaled = feature_scaler.transform(features)

    # Create sequence window
    X = np.repeat(row_scaled[:, None, :], WINDOW, axis=1)

    # LSTM embedding
    with torch.no_grad():
        emb = lstm.get_embedding(torch.tensor(X).float()).numpy()

    # TabNet prediction
    pred = tabnet.predict(emb.astype(np.float32))

    # Inverse transform to mg/dL
    pred = glucose_scaler.inverse_transform(
        pred.reshape(-1, 1)
    ).reshape(pred.shape)

    pred = np.clip(pred, 40, 400)[0]

    # ----------------------------
    # OUTPUT
    # ----------------------------
    labels = ["30 min", "60 min", "120 min"]

    st.success("Prediction Complete")

    for i, label in enumerate(labels):
        value = pred[i]

        st.metric(label, f"{value:.1f} mg/dL")

        if value < 80:
            st.success("🟢 Low glucose risk")
        elif value < 140:
            st.info("🟡 Normal range")
        elif value < 180:
            st.warning("🟠 Elevated")
        else:
            st.error("🔴 High risk")
