import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor

st.set_page_config(page_title="CGM AI Prediction", page_icon="🩸")

st.title("🩸 CGM AI Prediction System")
st.write("Predict blood glucose for the next 30, 60 and 120 minutes.")

# ============================================================
# LOAD MODELS
# ============================================================

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
                batch_first=True,
            )

            self.prediction_head = nn.Sequential(
                nn.Linear(hidden_size,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(64,n_horizons)
            )

        def forward(self,x):
            _,(h,_) = self.lstm(x)
            return self.prediction_head(h[-1])

        def get_embedding(self,x):
            _,(h,_) = self.lstm(x)
            return h[-1]

    lstm = LSTMWithPredictionHead(
        cfg["input_size"],
        cfg["hidden_size"],
        len(cfg["horizons"])
    )

    lstm.load_state_dict(
        torch.load(
            "lstm_encoder_trained.pth",
            map_location="cpu"
        )
    )

    lstm.eval()

    tabnet = TabNetRegressor()
    tabnet.load_model("tabnet_on_learned_embeddings.zip")

    return cfg, feature_scaler, glucose_scaler, lstm, tabnet


cfg, feature_scaler, glucose_scaler, lstm, tabnet = load_models()

WINDOW = cfg["window_size"]

# ============================================================
# USER INPUT
# ============================================================

st.header("Enter Current Information")

glucose = st.number_input(
    "Current Glucose (mg/dL)",
    40.0,
    400.0,
    110.0
)

heart_rate = st.number_input(
    "Heart Rate (bpm)",
    40.0,
    180.0,
    80.0
)

meal = st.radio(
    "Meal Eaten?",
    ["Yes","No"]
)

calories = st.number_input(
    "Calories",
    0.0,
    2000.0,
    250.0
)

carbs = st.number_input(
    "Carbohydrates (g)",
    0.0,
    300.0,
    40.0
)

protein = st.number_input(
    "Protein (g)",
    0.0,
    200.0,
    20.0
)

fat = st.number_input(
    "Fat (g)",
    0.0,
    200.0,
    10.0
)

fiber = st.number_input(
    "Fiber (g)",
    0.0,
    100.0,
    5.0
)

# ============================================================
# CREATE ALL 18 FEATURES AUTOMATICALLY
# ============================================================

meal_flag = 1 if meal=="Yes" else 0

hour = 12
day = 2

isnight = 1 if (hour<6 or hour>=22) else 0

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

# ============================================================
# PREDICT
# ============================================================

if st.button("Predict"):

    feature_order = cfg["feature_names"]

    row = np.array(
        [[features[f] for f in feature_order]],
        dtype=np.float32
    )

    row_scaled = feature_scaler.transform(row)

    X = np.repeat(
        row_scaled[:,None,:],
        WINDOW,
        axis=1
    )

    with torch.no_grad():

        embedding = lstm.get_embedding(
            torch.tensor(X,dtype=torch.float32)
        ).numpy()

    pred = tabnet.predict(
        embedding.astype(np.float32)
    )

    pred = glucose_scaler.inverse_transform(
        pred.reshape(-1,1)
    ).reshape(pred.shape)

    pred = np.clip(pred,40,400)[0]

    st.success("Prediction Complete")

    labels = ["30 Minutes","60 Minutes","120 Minutes"]

    for lbl,value in zip(labels,pred):

        st.subheader(lbl)

        st.metric(
            "Predicted Glucose",
            f"{value:.1f} mg/dL"
        )

        if value < 80:
            st.success("🟢 Low glucose")

        elif value < 140:
            st.warning("🟡 Normal glucose")

        elif value < 180:
            st.warning("🟠 Elevated glucose")

        else:
            st.error("🔴 High glucose")
