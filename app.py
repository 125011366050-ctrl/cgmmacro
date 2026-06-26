import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor

st.title("🩸 CGM AI Prediction System (LSTM + TabNet)")

@st.cache_resource
def load_models():
    cfg = joblib.load("model_config.pkl")
    feature_scaler = joblib.load("feature_scaler.pkl")
    glucose_scaler = joblib.load("glucose_scaler.pkl")

    class LSTMWithPredictionHead(nn.Module):
        def __init__(self, input_size, hidden_size, n_horizons, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                                 dropout=dropout, batch_first=True)
            self.prediction_head = nn.Sequential(
                nn.Linear(hidden_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, n_horizons)
            )
        def get_embedding(self, x):
            _, (h_n, _) = self.lstm(x)
            return h_n[-1]

    lstm = LSTMWithPredictionHead(cfg["input_size"], cfg["hidden_size"], len(cfg["horizons"]))
    lstm.load_state_dict(torch.load("lstm_encoder_trained.pth", map_location="cpu"))
    lstm.eval()

    tabnet = TabNetRegressor()
    tabnet.load_model("tabnet_on_learned_embeddings.zip")

    return cfg, feature_scaler, glucose_scaler, lstm, tabnet

cfg, feature_scaler, glucose_scaler, lstm, tabnet = load_models()
WINDOW = cfg["window_size"]
FEATS = cfg["feature_names"]

st.caption(f"Model expects {WINDOW} timesteps × {len(FEATS)} features.")
st.info("Enter your CURRENT reading. It will be repeated across the model's input window as an approximation — for real accuracy, feed the last actual sequence of readings instead.")

vals = {}
defaults = {"GL":100.0,"GL_MA_5":100.0,"GL_STD_5":5.0,"GL_Diff":0.0,"GL_Slope":0.0,
            "GL_Acceleration":0.0,"CV_Glucose":0.1,"HR":80.0,"HR_MA_5":80.0,"Meal_Flag":1,
            "Calories":200.0,"Carbs":30.0,"Protein":10.0,"Fat":5.0,"Fiber":3.0,
            "Hour":12,"DayOfWeek":2,"IsNight":0}
for f in FEATS:
    vals[f] = st.number_input(f, value=float(defaults.get(f, 0.0)))

if st.button("Predict Glucose"):
    row = np.array([[vals[f] for f in FEATS]], dtype=np.float32)
    row_scaled = feature_scaler.transform(row)
    X_seq = np.repeat(row_scaled[:, np.newaxis, :], WINDOW, axis=1)  # (1, WINDOW, n_features)

    with torch.no_grad():
        emb = lstm.get_embedding(torch.tensor(X_seq, dtype=torch.float32)).cpu().numpy()

    pred_scaled = tabnet.predict(emb.astype(np.float32))  # shape (1, 3)
    pred_mgdl = glucose_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    pred_mgdl = np.clip(pred_mgdl, 40, 400)[0]

    st.subheader("Result")
    labels = ["30 min", "60 min", "120 min"]
    for label, p in zip(labels, pred_mgdl):
        st.write(f"**{label}:** {p:.1f} mg/dL")
        if p < 80:
            st.success(f"{label}: Low Risk 🟢")
        elif p < 140:
            st.warning(f"{label}: Medium Risk 🟡")
        else:
            st.error(f"{label}: High Risk 🔴")