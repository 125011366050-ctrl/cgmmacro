import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    DATA_PATH = BASE_DIR
    FOOD_FILE = "Indian_Foods_GI_GL_Database.xlsx"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    HIDDEN_SIZE = 128
    N_HORIZONS = 3
    INPUT_SIZE = 18


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_horizons=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
        )
        self.output = nn.Linear(64, n_horizons)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        emb = self.embedding(h[-1])
        return self.output(emb)

    def get_embedding(self, x):
        _, (h, _) = self.lstm(x)
        return self.embedding(h[-1])


class PredictionEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.load_models()

    def load_models(self):
        print("Loading models...")
        base = self.config.DATA_PATH

        scaler_path = os.path.join(base, "glucose_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print("Scaler loaded")

        self.lstm = LSTMEncoder(
            input_size=self.config.INPUT_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            n_horizons=self.config.N_HORIZONS
        ).to(self.device)

        lstm_path = os.path.join(base, "lstm_encoder_trained.pth")
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Missing LSTM: {lstm_path}")

        state = torch.load(lstm_path, map_location=self.device)
        self.lstm.load_state_dict(state, strict=True)
        self.lstm.eval()
        print("LSTM loaded")

        from pytorch_tabnet.tab_model import TabNetRegressor
        self.tabnet = TabNetRegressor()
        tabnet_path = os.path.join(base, "tabnet_on_learned_embeddings.zip")
        if not os.path.exists(tabnet_path):
            raise FileNotFoundError(f"Missing TabNet: {tabnet_path}")
        self.tabnet.load_model(tabnet_path)
        print("TabNet loaded")

    def predict_glucose(self, x):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            emb = self.lstm.get_embedding(x).cpu().numpy()
        pred = self.tabnet.predict(emb)
        pred = self.scaler.inverse_transform(pred.reshape(-1, 1))
        return np.clip(pred.flatten(), 50, 400)


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


class ClinicalOrchestrator:
    def __init__(self, config):
        self.engine = PredictionEngine(config)
        self.risk = RiskEngine()

    def run(self, glucose, x):
        pred = self.engine.predict_glucose(x)
        risk = self.risk.compute(glucose, pred)
        return {
            "predictions": pred,
            "risk": risk
        }
