"""
CDSS SYSTEM - FIXED FULL VERSION
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
class Config:
   self.config.DATA_PATH = "."
    FOOD_FILE = "Indian_Foods_GI_GL_Database.xlsx"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    HIDDEN_SIZE = 128
    N_HORIZONS = 3


# =========================
# PATH HELPER (FIXED)
# =========================
def get_path(base_dir, file_name):
    return os.path.join(base_dir, file_name)


# =========================
# LSTM MODEL
# =========================
class LSTMWithPredictionHead(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_horizons=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_horizons)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

    def get_embedding(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# =========================
# PREDICTION ENGINE (FIXED)
# =========================
class PredictionEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.load_models()

    def load_models(self):
        print("Loading models...")

        base = self.config.DATA_PATH

        # ======================
        # SCALER FIXED
        # ======================
        scaler_path = get_path(base, "glucose_scaler.pkl")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")

        self.scaler = joblib.load(scaler_path)
        print("Scaler loaded")

        # ======================
        # TRAIN SHAPE
        # ======================
        X_train_path = get_path(base, "X_train.npy")

        if not os.path.exists(X_train_path):
            raise FileNotFoundError(f"Missing X_train: {X_train_path}")

        X_train = np.load(X_train_path)
        input_size = X_train.shape[2]

        print("Input size:", input_size)

        # ======================
        # LSTM
        # ======================
        self.lstm = LSTMWithPredictionHead(
            input_size=input_size,
            hidden_size=self.config.HIDDEN_SIZE,
            n_horizons=self.config.N_HORIZONS
        ).to(self.device)

        lstm_path = get_path(base, "lstm_encoder_trained.pth")

        self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
        self.lstm.eval()

        print("LSTM loaded")

        # ======================
        # TABNET
        # ======================
        from pytorch_tabnet.tab_model import TabNetRegressor

        self.tabnet = TabNetRegressor()
        tabnet_path = get_path(base, "tabnet_on_learned_embeddings.zip")

        self.tabnet.load_model(tabnet_path)

        print("TabNet loaded")

        self.embedding_dim = self.config.HIDDEN_SIZE

    # ======================
    # PREDICTION
    # ======================
    def predict_glucose(self, x):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]

        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            emb = self.lstm.get_embedding(x).cpu().numpy()

        pred = self.tabnet.predict(emb)

        pred = self.scaler.inverse_transform(pred.reshape(-1, 1))
        return np.clip(pred.flatten(), 50, 400)


# =========================
# RISK ENGINE
# =========================
class RiskEngine:
    def compute(self, current, pred):
        peak = np.max(pred)
        spike = peak - current

        if current > 200 or spike > 50:
            risk = "HIGH"
        elif spike > 20:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "risk": risk,
            "spike": float(spike),
            "peak": float(peak),
            "current": float(current)
        }


# =========================
# ORCHESTRATOR
# =========================
class ClinicalOrchestrator:
    def __init__(self, config):
        self.config = config
        self.engine = PredictionEngine(config)
        self.risk = RiskEngine()

    def run(self, glucose, x):
        pred = self.engine.predict_glucose(x)
        risk = self.risk.compute(glucose, pred)

        return {
            "predictions": pred,
            "risk": risk
        }


# =========================
# MAIN TEST
# =========================
def main():
    config = Config()
    system = ClinicalOrchestrator(config)

    X = np.load(os.path.join(config.DATA_PATH, "X_test.npy"))

    sample = X[0:1]

    result = system.run(120, sample)

    print("\nRESULT:")
    print(result)


if __name__ == "__main__":
    main()
