import streamlit as st
import numpy as np
import torch
import joblib

st.title("🩸 CGM AI Prediction System (LSTM + TabNet)")

# -------------------------
# LOAD SCALER + MODELS
# -------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("feature_scaler.pkl")

    # LSTM model
    class LSTMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=18, hidden_size=64, batch_first=True)
            self.fc = torch.nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
    lstm_model.eval()

    return scaler, lstm_model

scaler, lstm_model = load_models()

# -------------------------
# INPUTS (18 features)
# -------------------------
GL = st.number_input("GL", value=100.0)
GL_MA_5 = st.number_input("GL_MA_5", value=100.0)
GL_STD_5 = st.number_input("GL_STD_5", value=5.0)
GL_Diff = st.number_input("GL_Diff", value=0.0)

GL_Slope = st.number_input("GL_Slope", value=0.0)
GL_Acceleration = st.number_input("GL_Acceleration", value=0.0)
CV_Glucose = st.number_input("CV_Glucose", value=0.1)

HR = st.number_input("HR", value=80.0)
HR_MA_5 = st.number_input("HR_MA_5", value=80.0)

Meal_Flag = st.number_input("Meal_Flag", value=1)
Calories = st.number_input("Calories", value=200.0)
Carbs = st.number_input("Carbs", value=30.0)
Protein = st.number_input("Protein", value=10.0)
Fat = st.number_input("Fat", value=5.0)
Fiber = st.number_input("Fiber", value=3.0)

Hour = st.number_input("Hour", value=12)
DayOfWeek = st.number_input("DayOfWeek", value=2)
IsNight = st.number_input("IsNight", value=0)

# -------------------------
# PREDICT
# -------------------------
if st.button("Predict Glucose"):

    features = np.array([[
        GL, GL_MA_5, GL_STD_5, GL_Diff,
        GL_Slope, GL_Acceleration, CV_Glucose,
        HR, HR_MA_5,
        Meal_Flag, Calories, Carbs, Protein, Fat, Fiber,
        Hour, DayOfWeek, IsNight
    ]])

    # scale
    features_scaled = scaler.transform(features)

    # reshape for LSTM (1, time, features)
    X = features_scaled.reshape(1, 1, 18)

    # prediction
    with torch.no_grad():
        pred = lstm_model(torch.tensor(X, dtype=torch.float32)).item()

    st.subheader("Result")

    st.write(f"Predicted Glucose: {pred:.2f}")

    # risk logic
    if pred < 80:
        st.success("Low Risk 🟢")
    elif pred < 140:
        st.warning("Medium Risk 🟡")
    else:
        st.error("High Risk 🔴")