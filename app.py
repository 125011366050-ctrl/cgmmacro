import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetRegressor

st.set_page_config(page_title="CGM Recommendation System", page_icon="🩸")
st.title("🩸 CGM AI Prediction + Recommendation System")
st.write("Predict glucose for next 30, 60, 120 minutes + get diet advice")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    cfg = joblib.load("model_config.pkl")
    feature_scaler = joblib.load("feature_scaler.pkl")
    glucose_scaler = joblib.load("glucose_scaler.pkl")

    class LSTMWithPredictionHead(nn.Module):
        def __init__(self, input_size, hidden_size, n_horizons, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout,
                batch_first=True
            )

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return h[-1]

        def get_embedding(self, x):
            _, (h, _) = self.lstm(x)
            return h[-1]

    lstm = LSTMWithPredictionHead(
        cfg["input_size"],
        cfg["hidden_size"],
        len(cfg["horizons"])
    )

    lstm.load_state_dict(torch.load("lstm_encoder_trained.pth", map_location="cpu"))
    lstm.eval()

    tabnet = TabNetRegressor()
    tabnet.load_model("tabnet_on_learned_embeddings.zip")

    return cfg, feature_scaler, glucose_scaler, lstm, tabnet


cfg, feature_scaler, glucose_scaler, lstm, tabnet = load_models()
WINDOW = cfg["window_size"]

# -----------------------------
# USER INPUT
# -----------------------------
st.header("Enter Patient Details")

glucose = st.number_input("Current Glucose (mg/dL)", 40.0, 400.0, 110.0)
heart_rate = st.number_input("Heart Rate (bpm)", 40.0, 180.0, 80.0)
meal = st.radio("Meal Eaten?", ["Yes", "No"])
calories = st.number_input("Calories", 0.0, 2000.0, 250.0)
carbs = st.number_input("Carbohydrates (g)", 0.0, 300.0, 40.0)
protein = st.number_input("Protein (g)", 0.0, 200.0, 20.0)
fat = st.number_input("Fat (g)", 0.0, 200.0, 10.0)
fiber = st.number_input("Fiber (g)", 0.0, 100.0, 5.0)

meal_flag = 1 if meal == "Yes" else 0
now = datetime.now()
hour = now.hour
day = now.weekday()
isnight = 1 if (hour < 6 or hour >= 22) else 0

# -----------------------------
# FEATURE VECTOR
# -----------------------------
features = {
    "GL": glucose,
    "GL_MA_5": glucose,
    "GL_STD_5": 5.0,
    "GL_Diff": 0.0,
    "GL_Slope": 0.0,
    "GL_Acceleration": 0.0,
    "CV_Glucose": 0.10,
    "HR": heart_rate,
    "HR_MA_5": heart_rate,
    "Meal_Flag": meal_flag,
    "Calories": calories,
    "Carbs": carbs,
    "Protein": protein,
    "Fat": fat,
    "Fiber": fiber,
    "Hour": hour,
    "DayOfWeek": day,
    "IsNight": isnight,
}

feature_order = cfg["feature_names"]

# -----------------------------
# RECOMMENDATION SYSTEM
# -----------------------------
def recommend_food(pred):
    g30, g60, g120 = pred

    if g120 > 180:
        return [
            "🚨 High risk detected",
            "Eat low GI foods: oats, green vegetables, legumes",
            "Avoid rice, sugar, sweets, processed carbs",
            "Take light walk after meal"
        ]

    elif g120 > 140:
        return [
            "⚠️ Moderate glucose rise expected",
            "Eat balanced meal: brown rice + protein + fiber",
            "Avoid sugary snacks"
        ]

    else:
        return [
            "✅ Stable glucose predicted",
            "Normal balanced diet is fine",
            "Include protein + fiber for stability"
        ]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict & Recommend"):

    row = np.array([[features[f] for f in feature_order]], dtype=np.float32)
    row_scaled = feature_scaler.transform(row)

    X = np.repeat(row_scaled[:, None, :], WINDOW, axis=1)

    with torch.no_grad():
        embedding = lstm.get_embedding(
            torch.tensor(X, dtype=torch.float32)
        ).numpy()

    pred_scaled = tabnet.predict(embedding.astype(np.float32))
    pred = glucose_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(pred_scaled.shape)

    pred = np.clip(pred, 40, 400)[0]

    labels = ["30 min", "60 min", "120 min"]

    st.subheader("🧠 Glucose Prediction")

    for l, v in zip(labels, pred):
        st.metric(l, f"{v:.1f} mg/dL")

    # -----------------------------
    # SHOW RECOMMENDATION
    # -----------------------------
    st.subheader("🍎 Personalized Recommendation")

    reco = recommend_food(pred)

    for r in reco:
        st.write("•", r)
