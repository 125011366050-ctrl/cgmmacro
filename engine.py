import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import warnings
from datetime import datetime
from typing import Dict, Tuple
from dataclasses import dataclass
from pytorch_tabnet.tab_model import TabNetRegressor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    DATA_PATH: str = BASE_DIR
    FOOD_FILE: str = os.path.join(BASE_DIR, "Indian_Foods_GI_GL_Database (1).xlsx")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42
    HIDDEN_SIZE: int = 128
    N_HORIZONS: int = 3
    INPUT_SIZE: int = 18
    WINDOW_SIZE: int = 36
    LOW_SPIKE: float = 20.0
    MEDIUM_SPIKE: float = 50.0
    CRITICAL_GLUCOSE: float = 200.0
    WARNING_GLUCOSE: float = 180.0
    HIGH_RISK_GI_MAX: float = 40.0
    HIGH_RISK_GL_MAX: float = 15.0
    MEDIUM_RISK_GI_MAX: float = 55.0
    MEDIUM_RISK_GL_MAX: float = 20.0
    SOFT_MAX_STEP: float = 60.0
    PHYSIO_NOISE_STD: float = 2.0
    UNCERTAINTY_STD: Tuple[float, float, float] = (8.0, 12.0, 15.0)
    HYPO_THRESHOLD: float = 70.0
    HYPO_WARNING: float = 90.0
    CRITICAL_DROP: float = 50.0
    MODERATE_DROP: float = 35.0
    DROP_HIGH_ALERT: float = 70.0
    DROP_MODERATE_ALERT: float = 50.0
    DROP_CAUTION: float = 35.0
    VELOCITY_HIGH_RISK: float = 3.0
    VELOCITY_MEDIUM_RISK: float = 1.5


# ─────────────────────────────────────────────────────────────
#  REMOTE MODEL DOWNLOAD (Google Drive fallback)
# ─────────────────────────────────────────────────────────────
# Fill in your real Google Drive file IDs below.
# To get a file ID: share the file on Drive -> "Anyone with link" ->
# the ID is the long string in the URL between /d/ and /view
#   https://drive.google.com/file/d/FILE_ID_HERE/view

REMOTE_FILES = {
    "feature_scaler.pkl":                  "PUT_FEATURE_SCALER_FILE_ID_HERE",
    "glucose_scaler.pkl":                  "PUT_GLUCOSE_SCALER_FILE_ID_HERE",
    "lstm_encoder_trained.pth":             "PUT_LSTM_FILE_ID_HERE",
    "tabnet_on_learned_embeddings.zip":     "PUT_TABNET_FILE_ID_HERE",
    "embedding_scaler.pkl":                 "PUT_EMBEDDING_SCALER_FILE_ID_HERE",  # optional
    "Indian_Foods_GI_GL_Database (1).xlsx": "PUT_FOOD_DB_FILE_ID_HERE",
}


def ensure_file(filename: str, base_dir: str, required: bool = True) -> str:
    """
    Returns the local path to `filename` inside base_dir.
    If missing locally and a Drive ID is configured, downloads it first.
    """
    local_path = os.path.join(base_dir, filename)
    if os.path.exists(local_path):
        return local_path

    file_id = REMOTE_FILES.get(filename, "")
    placeholder = file_id.startswith("PUT_") or not file_id

    if placeholder:
        if required:
            raise FileNotFoundError(
                f"Missing file: {local_path}\n"
                f"It is not in the repo AND no valid Google Drive ID is set "
                f"in REMOTE_FILES['{filename}']. Either commit the file to "
                f"the repo, or put a real Drive file ID in REMOTE_FILES."
            )
        else:
            print(f"⚠️ Optional file '{filename}' not found and not configured — skipping.")
            return local_path  # caller is expected to check os.path.exists again

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is not installed. Add 'gdown' to requirements.txt to enable "
            "automatic model downloads."
        )

    print(f"⬇️ Downloading '{filename}' from Google Drive (id={file_id})...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, local_path, quiet=False)

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Download of '{filename}' failed — file still missing at {local_path}. "
            f"Check that the Drive file ID is correct and shared as 'Anyone with link'."
        )
    print(f"✓ Downloaded '{filename}'")
    return local_path


def build_cgm_features(g: np.ndarray, carbs=0, protein=0, fat=0) -> np.ndarray:
    g = np.array(g, dtype=np.float32)
    if len(g) < 3:
        g = np.pad(g, (3 - len(g), 0), mode='edge')
    current = g[-1]
    mean_g = np.mean(g)
    std_g = np.std(g)
    slope = float(np.polyfit(np.arange(len(g)), g, 1)[0])
    acceleration = float(np.mean(np.diff(np.diff(g)))) if len(g) >= 3 else 0.0
    gl_ma_5 = float(np.mean(g[-5:])) if len(g) >= 5 else mean_g
    gl_std_5 = float(np.std(g[-5:])) if len(g) >= 5 else std_g
    spike = float(np.max(g) - np.mean(g[:3])) if len(g) >= 3 else 0.0
    gl_diff = float(g[-1] - g[-2]) if len(g) >= 2 else 0.0
    cv = float(std_g / (mean_g + 1e-6))
    roc = float((g[-1] - g[-5]) / 5) if len(g) >= 6 else slope
    GL = current
    time_of_day = 12.0
    is_post_meal = 1.0
    activity_level = 0.0
    return np.array([
        GL, gl_ma_5, gl_std_5, gl_diff, slope,
        acceleration, cv, current, mean_g,
        1.0, float(carbs), float(protein), float(fat),
        spike, roc, time_of_day, is_post_meal, activity_level
    ], dtype=np.float32)


def build_lstm_input(cgm_readings, carbs=0, protein=0, fat=0, window_size=36) -> np.ndarray:
    cgm = np.array(cgm_readings, dtype=np.float32)
    if len(cgm) < 10:
        cgm = np.pad(cgm, (10 - len(cgm), 0), mode='edge')
    cgm_interp = np.interp(
        np.linspace(0, len(cgm) - 1, window_size),
        np.arange(len(cgm)), cgm
    )
    sequence = []
    for t in range(window_size):
        end = t + 1
        start = max(0, end - 10)
        window = cgm_interp[start:end]
        if len(window) < 10:
            window = np.pad(window, (10 - len(window), 0), mode='edge')
        features = build_cgm_features(window, carbs=carbs, protein=protein, fat=fat)
        sequence.append(features)
    seq = np.array(sequence, dtype=np.float32)
    assert seq.shape == (window_size, 18), f"Sequence shape mismatch: {seq.shape}"
    return seq


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_horizons=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(64, n_horizons)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.output(self.embedding(h[-1]))

    def get_embedding(self, x):
        _, (h, _) = self.lstm(x)
        return self.embedding(h[-1])


def _probe_scaler(scaler, n_horizons=3):
    try:
        scaler.inverse_transform(np.zeros((1, n_horizons)))
        return 'multi'
    except Exception:
        pass
    try:
        scaler.inverse_transform(np.zeros((1, 1)))
        return 'single'
    except Exception:
        pass
    return 'none'


def safe_risk(risk: dict) -> dict:
    """Crash-proof accessor — works with both old and new key names."""
    return {
        "risk_level":       risk.get("risk_level", "LOW"),
        "spike":            risk.get("upward_spike", risk.get("spike", 0.0)),
        "upward_spike":     risk.get("upward_spike", risk.get("spike", 0.0)),
        "drop":             risk.get("downward_drop", risk.get("drop", 0.0)),
        "downward_drop":    risk.get("downward_drop", risk.get("drop", 0.0)),
        "drop_severity":    risk.get("drop_severity", "NORMAL"),
        "trend":            risk.get("trend", 0.0),
        "trend_direction":  risk.get("trend_direction", "STABLE"),
        "peak":             risk.get("peak", 0.0),
        "trough":           risk.get("trough", 0.0),
        "current":          risk.get("current", 0.0),
        "hypo_risk":        risk.get("hypo_risk", False),
        "hypo_warning":     risk.get("hypo_warning", False),
        "predictions":      risk.get("predictions", []),
        "requires_action":  risk.get("requires_action", False),
        "glucose_velocity": risk.get("glucose_velocity", 0.0),
        "velocity_risk":    risk.get("velocity_risk", "STABLE"),
        "dominant_risk":    risk.get("dominant_risk", "NONE"),
        "risk_score":       risk.get("risk_score", 0.0),
        "clinical_summary": risk.get("clinical_summary", ""),
        "uncertainty":      risk.get("uncertainty", {"lower": [], "upper": [], "std": []}),
    }


def soft_physiology_adjust(pred: np.ndarray, current: float,
                            max_step: float = 60.0) -> np.ndarray:
    pred = pred.copy().astype(np.float32)
    all_points = np.insert(pred, 0, current)
    diffs = np.diff(all_points)
    soft_diffs = np.tanh(diffs / max_step) * max_step
    pred = current + np.cumsum(soft_diffs)
    return pred.astype(np.float32)


def add_uncertainty(pred: np.ndarray,
                    std: Tuple[float, float, float] = (8.0, 12.0, 15.0)
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    std_arr = np.array(std[:len(pred)], dtype=np.float32)
    lower = pred - 1.96 * std_arr
    upper = pred + 1.96 * std_arr
    lower = np.clip(lower, 40, 400)
    upper = np.clip(upper, 40, 400)
    return pred, lower, upper


def compute_true_peak(current: float, pred: np.ndarray) -> float:
    all_points = np.append(float(current), pred.astype(float))
    return float(np.max(all_points))


def compute_true_trough(current: float, pred: np.ndarray) -> float:
    all_points = np.append(float(current), pred.astype(float))
    return float(np.min(all_points))


class PredictionEngine:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.rng = np.random.default_rng(seed=config.SEED)
        self.load_models()

    def load_models(self):
        base = self.config.DATA_PATH

        feature_scaler_path = ensure_file("feature_scaler.pkl", base, required=True)
        self.feature_scaler = joblib.load(feature_scaler_path)
        print("✓ Feature scaler loaded")

        scaler_path = ensure_file("glucose_scaler.pkl", base, required=True)
        self.scaler = joblib.load(scaler_path)
        self.scaler_mode = _probe_scaler(self.scaler, self.config.N_HORIZONS)
        print(f"✓ Glucose scaler loaded — mode: {self.scaler_mode}")

        self.lstm = LSTMEncoder(
            input_size=self.config.INPUT_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            n_horizons=self.config.N_HORIZONS
        ).to(self.device)

        lstm_path = ensure_file("lstm_encoder_trained.pth", base, required=True)
        self.lstm.load_state_dict(
            torch.load(lstm_path, map_location=self.device),
            strict=False
        )
        self.lstm.eval()
        for param in self.lstm.parameters():
            param.requires_grad = False
        print("✓ LSTM loaded and frozen")

        tabnet_path = ensure_file("tabnet_on_learned_embeddings.zip", base, required=True)
        self.tabnet = TabNetRegressor()
        self.tabnet.load_model(tabnet_path)
        print("✓ TabNet loaded")

        embedding_scaler_path = ensure_file("embedding_scaler.pkl", base, required=False)
        if os.path.exists(embedding_scaler_path):
            self.embedding_scaler = joblib.load(embedding_scaler_path)
            print("✓ Embedding scaler loaded")
        else:
            self.embedding_scaler = None
            print("⚠️ No embedding scaler — using z-norm fallback")

    def _scale_features(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        x_scaled = self.feature_scaler.transform(x_flat)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return x_scaled.reshape(original_shape)

    def _inverse_scale(self, raw: np.ndarray) -> np.ndarray:
        raw = np.array(raw, dtype=np.float32).reshape(-1)
        try:
            if self.scaler_mode == 'multi':
                result = self.scaler.inverse_transform(raw.reshape(1, -1)).flatten()
            elif self.scaler_mode == 'single':
                result = np.array([
                    self.scaler.inverse_transform([[v]])[0][0] for v in raw
                ])
            else:
                result = raw.copy()
            result = np.nan_to_num(result, nan=120.0, posinf=400.0, neginf=40.0)
            return result
        except Exception as e:
            print(f"⚠️ inverse_scale failed: {e} — returning raw")
            return raw.copy()

    def predict_glucose(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        current_glucose = float(x[0, -1, 0])

        x_scaled = self._scale_features(x)
        t = torch.from_numpy(x_scaled).float().to(self.device)
        with torch.no_grad():
            emb = self.lstm.get_embedding(t).cpu().numpy().astype(np.float32)
        emb = emb.reshape(emb.shape[0], -1)

        if self.embedding_scaler is not None:
            emb = self.embedding_scaler.transform(emb)
        else:
            emb_mean = np.mean(emb, axis=1, keepdims=True)
            emb_std = np.std(emb, axis=1, keepdims=True)
            emb = (emb - emb_mean) / (emb_std + 1e-6)
            emb = np.clip(emb, -5, 5)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        raw = self.tabnet.predict(emb)
        raw = np.array(raw, dtype=np.float32).flatten()

        pred = self._inverse_scale(raw)
        pred = np.clip(pred, 40, 400)

        if len(pred) == 1:
            base_val = float(pred[0])
            pred = np.array([base_val, base_val * 1.01, base_val * 1.02], dtype=np.float32)

        pred = soft_physiology_adjust(pred, current_glucose, self.config.SOFT_MAX_STEP)
        pred = np.clip(pred, 40, 400)

        mean_pred, lower, upper = add_uncertainty(pred, self.config.UNCERTAINTY_STD)

        return mean_pred, lower, upper

    def _classify_drop(self, drop: float) -> str:
        if drop >= self.config.DROP_HIGH_ALERT:
            return "HIGH_ALERT"
        elif drop >= self.config.DROP_MODERATE_ALERT:
            return "MODERATE_ALERT"
        elif drop >= self.config.DROP_CAUTION:
            return "CAUTION"
        else:
            return "NORMAL"

    def _compute_glucose_velocity(self, current: float, pred: np.ndarray) -> float:
        return float((pred[0] - current) / 30.0)

    def _classify_velocity(self, velocity: float) -> str:
        abs_v = abs(velocity)
        if abs_v >= self.config.VELOCITY_HIGH_RISK:
            return "RAPID_FALL" if velocity < 0 else "RAPID_RISE"
        elif abs_v >= self.config.VELOCITY_MEDIUM_RISK:
            return "MODERATE_FALL" if velocity < 0 else "MODERATE_RISE"
        return "STABLE"

    def _compute_risk_score(self, current: float, upward_spike: float,
                             downward_drop: float, trough: float,
                             velocity: float) -> float:
        hyper_score    = min(1.0, max(0.0, (current - 140) / 100.0))
        spike_score    = min(1.0, upward_spike / 80.0)
        drop_score     = min(1.0, downward_drop / 80.0)
        trough_score   = min(1.0, max(0.0, (90 - trough) / 50.0))
        velocity_score = min(1.0, abs(velocity) / self.config.VELOCITY_HIGH_RISK)
        score = (
            0.20 * hyper_score +
            0.20 * spike_score +
            0.25 * drop_score +
            0.20 * trough_score +
            0.15 * velocity_score
        )
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    def _build_clinical_summary(self, risk_level: str, dominant_risk: str,
                                  current: float, trough: float,
                                  upward_spike: float, downward_drop: float,
                                  velocity: float, hypo_risk: bool,
                                  hypo_warning: bool) -> str:
        direction = "falling" if velocity < 0 else "rising"
        rate = abs(round(velocity, 2))
        if dominant_risk == "HYPOGLYCEMIA":
            return (f"HYPOGLYCEMIA ALERT: Glucose {current:.0f} mg/dL falling at "
                    f"{rate} mg/dL/min. Predicted trough {trough:.0f} mg/dL — "
                    f"below safe threshold. Immediate action required.")
        elif dominant_risk == "DROP_RISK":
            return (f"FALL RISK: Glucose {current:.0f} mg/dL {direction} at "
                    f"{rate} mg/dL/min. Predicted drop of {downward_drop:.0f} mg/dL "
                    f"to {trough:.0f} mg/dL over 2 hours.")
        elif dominant_risk == "HYPERGLYCEMIA":
            return (f"HIGH GLUCOSE: Current {current:.0f} mg/dL with predicted spike "
                    f"of {upward_spike:.0f} mg/dL. Glucose control intervention needed.")
        elif dominant_risk == "HYPO_WARNING":
            return (f"EARLY WARNING: Glucose {current:.0f} mg/dL trending {direction}. "
                    f"Predicted to reach {trough:.0f} mg/dL — approaching caution zone.")
        else:
            return (f"Glucose {current:.0f} mg/dL — {direction} at {rate} mg/dL/min. "
                    f"Within acceptable range. Continue monitoring.")

    def compute_risk(self, current: float,
                     predictions: np.ndarray,
                     lower: np.ndarray,
                     upper: np.ndarray) -> Dict:
        peak   = compute_true_peak(current, predictions)
        trough = compute_true_trough(current, predictions)

        worst_trough = compute_true_trough(current, lower)
        worst_peak   = compute_true_peak(current, upper)

        upward_spike  = max(0.0, worst_peak - current)
        downward_drop = max(0.0, current - worst_trough)

        drop_severity  = self._classify_drop(downward_drop)
        trend          = float(predictions[0] - current)
        trend_direction = "FALLING" if trend < -5 else "RISING" if trend > 5 else "STABLE"

        velocity      = self._compute_glucose_velocity(current, predictions)
        velocity_risk = self._classify_velocity(velocity)

        hypo_risk    = worst_trough < self.config.HYPO_THRESHOLD
        hypo_warning = worst_trough < self.config.HYPO_WARNING

        drop_risk_high   = downward_drop >= self.config.CRITICAL_DROP
        drop_risk_medium = downward_drop >= self.config.MODERATE_DROP
        spike_risk_high  = upward_spike  >= self.config.MEDIUM_SPIKE
        spike_risk_medium = upward_spike >= self.config.LOW_SPIKE
        velocity_high    = abs(velocity) >= self.config.VELOCITY_HIGH_RISK

        if (current >= self.config.CRITICAL_GLUCOSE or hypo_risk or
                drop_risk_high or spike_risk_high or velocity_high):
            risk_level = "HIGH"
        elif (current >= self.config.WARNING_GLUCOSE or hypo_warning or
              drop_risk_medium or spike_risk_medium or
              drop_severity in ("MODERATE_ALERT", "CAUTION")):
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if hypo_risk:
            dominant_risk = "HYPOGLYCEMIA"
        elif drop_risk_high or drop_severity == "HIGH_ALERT":
            dominant_risk = "DROP_RISK"
        elif current >= self.config.CRITICAL_GLUCOSE or spike_risk_high:
            dominant_risk = "HYPERGLYCEMIA"
        elif hypo_warning or drop_risk_medium:
            dominant_risk = "HYPO_WARNING"
        else:
            dominant_risk = "NONE"

        risk_score = self._compute_risk_score(
            current, upward_spike, downward_drop, trough, velocity
        )
        clinical_summary = self._build_clinical_summary(
            risk_level, dominant_risk, current, trough,
            upward_spike, downward_drop, velocity, hypo_risk, hypo_warning
        )

        return {
            "risk_level":       risk_level,
            "risk_score":       risk_score,
            "dominant_risk":    dominant_risk,
            "clinical_summary": clinical_summary,
            "current":          float(current),
            "peak":             peak,
            "trough":           trough,
            "worst_peak":       worst_peak,
            "worst_trough":     worst_trough,
            "upward_spike":     upward_spike,
            "spike":            upward_spike,
            "downward_drop":    downward_drop,
            "drop":             downward_drop,
            "drop_severity":    drop_severity,
            "trend":            trend,
            "trend_direction":  trend_direction,
            "glucose_velocity": round(velocity, 3),
            "velocity_risk":    velocity_risk,
            "hypo_risk":        hypo_risk,
            "hypo_warning":     hypo_warning,
            "predictions":      predictions.tolist(),
            "uncertainty": {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "std":   list(self.config.UNCERTAINTY_STD),
            },
            "requires_action": risk_level in ["MEDIUM", "HIGH"],
        }


def load_food_database(food_file: str) -> pd.DataFrame:
    for sheet in ["GI & Nutrition Data", "Sheet1", 0]:
        try:
            df = pd.read_excel(food_file, sheet_name=sheet)
            print(f"✓ Food DB loaded — sheet: {sheet} | rows: {len(df)}")
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(f"Cannot read food file: {food_file}")

    df.columns = df.columns.str.strip()
    rename_map = {
        "Food Name": "Food_Name", "food name": "Food_Name",
        "FoodName": "Food_Name", "Item": "Food_Name", "Name": "Food_Name",
        "Carbs (g)": "Carbs", "Carbohydrates (g)": "Carbs",
        "Carbohydrates": "Carbs", "carbs": "Carbs", "CHO (g)": "Carbs",
        "Protein (g)": "Protein", "protein": "Protein", "PROTEIN": "Protein",
        "Fat (g)": "Fat", "fat": "Fat", "Total Fat (g)": "Fat",
        "Calories (kcal)": "Calories", "Energy (kcal)": "Calories",
        "Kcal": "Calories", "kcal": "Calories", "Energy": "Calories",
        "Fiber (g)": "Fiber", "Dietary Fiber (g)": "Fiber",
        "Dietary Fiber": "Fiber", "fiber": "Fiber",
        "gi": "GI", "gl": "GL",
        "Glycemic Index": "GI", "Glycemic Load": "GL",
        "category": "Category", "CATEGORY": "Category", "Type": "Category",
        "Serving (g)": "Serving", "Serving Size": "Serving"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    defaults = {
        "Food_Name": "Unknown", "GI": 55, "GL": 10,
        "Carbs": 30, "Protein": 5, "Fat": 5,
        "Calories": 200, "Fiber": 0, "Category": "General"
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    for col in ["GI", "GL", "Carbs", "Protein", "Fat", "Calories", "Fiber"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["GI"] = df["GI"].replace(0, np.nan)
    median_gi = df["GI"].median()
    df["GI"] = df["GI"].fillna(median_gi)
    print(f"✓ Food DB ready | GI range: {df['GI'].min():.0f}–{df['GI'].max():.0f} | median GI: {median_gi:.0f}")
    return df


class FoodRankingEngine:
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8) if mx > mn else np.ones_like(x) * 0.5

    def estimate_spike(self, food: dict, current_glucose: float) -> float:
        gi = float(food.get("GI", 55))
        carbs = float(food.get("Carbs", 30))
        fiber = float(food.get("Fiber", 0))
        protein = float(food.get("Protein", 5))
        effective_carbs = max(0.0, carbs - fiber * 0.5)
        spike = (gi / 100.0) * (effective_carbs / 50.0) * 40.0
        spike *= max(0.7, 1.0 - protein * 0.01)
        return float(np.clip(spike, 0, 200))

    def filter_by_risk(self, df: pd.DataFrame, risk_level: str,
                        dominant_risk: str = "NONE") -> pd.DataFrame:
        if dominant_risk == "HYPOGLYCEMIA":
            filtered = df[df["GI"] >= 55]
            if len(filtered) < 5:
                filtered = df.nlargest(20, "GI")
        elif dominant_risk in ("DROP_RISK", "HYPO_WARNING"):
            filtered = df[(df["GI"] >= 40) & (df["GI"] <= 70)]
            if len(filtered) < 5:
                filtered = df[(df["GI"] >= 35) & (df["GI"] <= 75)]
        elif risk_level == "HIGH":
            filtered = df[df["GI"] <= 40]
            if len(filtered) < 5:
                filtered = df[df["GI"] <= 55]
        elif risk_level == "MEDIUM":
            filtered = df[df["GI"] <= 55]
            if len(filtered) < 5:
                filtered = df[df["GI"] <= 70]
        else:
            filtered = df.copy()
        if len(filtered) < 5:
            filtered = df.nsmallest(50, "GI")
        return filtered.reset_index(drop=True)

    def rank(self, df: pd.DataFrame, risk_level: str,
             current_glucose: float, top_k: int = 10,
             dominant_risk: str = "NONE") -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        filtered = self.filter_by_risk(df, risk_level, dominant_risk)
        out = filtered.copy()
        spikes = np.array([
            self.estimate_spike(row.to_dict(), current_glucose)
            for _, row in out.iterrows()
        ], dtype=float)
        out["Predicted_Spike"] = spikes
        out["Predicted_Peak"] = spikes + current_glucose
        spike_n = self._normalize(spikes)
        gi_n    = self._normalize(out["GI"].values)
        gl_n    = self._normalize(out["GL"].values)
        prot_n  = self._normalize(out["Protein"].values)
        fiber_n = self._normalize(out["Fiber"].values)

        if dominant_risk == "HYPOGLYCEMIA":
            w = dict(spike=0.05, gi=0.10, gl=0.10, protein=0.35, fiber=0.40)
        elif dominant_risk in ("DROP_RISK", "HYPO_WARNING"):
            w = dict(spike=0.10, gi=0.15, gl=0.15, protein=0.30, fiber=0.30)
        elif risk_level == "HIGH":
            w = dict(spike=0.35, gi=0.25, gl=0.20, protein=0.10, fiber=0.10)
        elif risk_level == "MEDIUM":
            w = dict(spike=0.25, gi=0.25, gl=0.20, protein=0.15, fiber=0.15)
        else:
            w = dict(spike=0.15, gi=0.20, gl=0.15, protein=0.25, fiber=0.25)

        out["Score"] = (
            w["spike"] * (1 - spike_n) +
            w["gi"]    * (1 - gi_n)    +
            w["gl"]    * (1 - gl_n)    +
            w["protein"] * prot_n      +
            w["fiber"]   * fiber_n
        )
        out = out.sort_values("Score", ascending=False).head(top_k).reset_index(drop=True)
        out["Rank"] = range(1, len(out) + 1)
        out["Recommendation"] = out["Score"].apply(
            lambda s: "⭐ Top Pick" if s > 0.75
            else "👍 Good Choice" if s > 0.55
            else "✓ Acceptable"
        )
        return out

    def meal_plan(self, df: pd.DataFrame, risk_level: str,
                  current_glucose: float, dominant_risk: str = "NONE") -> Dict:
        meal_keywords = {
            "Breakfast": ["Breakfast", "Idli", "Dosa", "Porridge", "Oats", "Upma"],
            "Lunch":     ["Rice", "Dal", "Curry", "Wheat", "Lentil", "Sabzi"],
            "Dinner":    ["Roti", "Wheat", "Millet", "Curry", "Vegetable", "Dal"],
            "Snack":     ["Snack", "Fruit", "Nut", "Seed", "Salad", "Egg", "Dairy"]
        }
        plan = {}
        for meal, keywords in meal_keywords.items():
            pattern = "|".join(keywords)
            if "Category" in df.columns:
                subset = df[df["Category"].str.contains(pattern, case=False, na=False)]
            else:
                subset = pd.DataFrame()
            if len(subset) < 3:
                subset = df.nsmallest(50, "GI")
            ranked = self.rank(subset, risk_level, current_glucose,
                               top_k=3, dominant_risk=dominant_risk)
            if not ranked.empty:
                plan[meal] = ranked[["Food_Name", "GI", "GL",
                                      "Predicted_Spike", "Score"]].to_dict("records")
            else:
                plan[meal] = []
        return plan


class ActivityEngine:
    def recommend(self, risk_info: Dict) -> Dict:
        r = safe_risk(risk_info)
        risk       = r["risk_level"]
        glucose    = r["current"]
        upspike    = r["upward_spike"]
        hypo       = r["hypo_risk"]
        hypo_warn  = r["hypo_warning"]
        drop_sev   = r["drop_severity"]
        dominant   = r["dominant_risk"]

        if hypo:
            return {
                "activity":       "REST — Do NOT exercise",
                "duration":       "Until glucose > 90 mg/dL",
                "timing":         "IMMEDIATELY check glucose",
                "intensity":      "None",
                "calorie_burn":   "0 kcal",
                "clinical_alert": "🚨 HYPOGLYCEMIA RISK — Consume fast-acting carbs NOW",
                "evidence":       "Exercise contraindicated below 70 mg/dL",
                "urgency_score":  1.0
            }
        if dominant == "DROP_RISK" or drop_sev == "HIGH_ALERT":
            return {
                "activity":       "REST — Sit down immediately",
                "duration":       "20–30 min, recheck glucose every 15 min",
                "timing":         "NOW",
                "intensity":      "None",
                "calorie_burn":   "0 kcal",
                "clinical_alert": "🚨 SEVERE DROP predicted — hypoglycemia risk imminent",
                "evidence":       "Drop >70 mg/dL in 2h — immediate monitoring required",
                "urgency_score":  0.95
            }
        if hypo_warn or drop_sev == "MODERATE_ALERT":
            return {
                "activity":       "Seated Rest / Light Stretching only",
                "duration":       "15–20 minutes",
                "timing":         "Monitor glucose every 15 min",
                "intensity":      "Very Low",
                "calorie_burn":   "20–40 kcal",
                "clinical_alert": "⚠️ MODERATE DROP predicted — avoid strenuous activity",
                "evidence":       "Drop >50 mg/dL — caution advised",
                "urgency_score":  0.7
            }
        if drop_sev == "CAUTION":
            return {
                "activity":       "Light Walking only",
                "duration":       "10–15 minutes",
                "timing":         "After confirming glucose is stable",
                "intensity":      "Low",
                "calorie_burn":   "40–60 kcal",
                "clinical_alert": "⚠️ Mild drop predicted — light activity only",
                "evidence":       "Drop >35 mg/dL — monitor trend",
                "urgency_score":  0.4
            }
        if glucose >= 200 or upspike > 60:
            return {
                "activity":       "URGENT — Brisk Walking",
                "duration":       "40–45 minutes",
                "timing":         "IMMEDIATELY within 10 minutes",
                "intensity":      "High",
                "calorie_burn":   "200–250 kcal",
                "clinical_alert": "🚨 CRITICAL HIGH — Do not remain sedentary",
                "evidence":       "Emergency glucose reduction protocol",
                "urgency_score":  1.0
            }
        if risk == "HIGH":
            return {
                "activity":       "Brisk Walking / Aerobic Exercise",
                "duration":       "30–45 minutes",
                "timing":         "Within 15 min after meals",
                "intensity":      "Moderate to High",
                "calorie_burn":   "180–250 kcal",
                "clinical_alert": "⚠️ High risk — Activity essential",
                "evidence":       "Reduces post-meal spike by 30–40%",
                "urgency_score":  0.8
            }
        if risk == "MEDIUM":
            return {
                "activity":       "Brisk Walking / Cycling",
                "duration":       "20–30 minutes",
                "timing":         "30–45 min after meals",
                "intensity":      "Moderate",
                "calorie_burn":   "120–180 kcal",
                "clinical_alert": "Activity recommended",
                "evidence":       "Reduces post-meal glucose by 20–30%",
                "urgency_score":  0.5
            }
        return {
            "activity":       "Light Walking / Yoga",
            "duration":       "10–15 minutes",
            "timing":         "Any time",
            "intensity":      "Low",
            "calorie_burn":   "50–80 kcal",
            "clinical_alert": "Maintain regular activity",
            "evidence":       "Maintains baseline insulin sensitivity",
            "urgency_score":  0.2
        }


class ClinicalOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.prediction_engine = PredictionEngine(config)
        food_path = ensure_file(
            "Indian_Foods_GI_GL_Database (1).xlsx", config.DATA_PATH, required=True
        )
        self.food_df = load_food_database(food_path)
        self.ranking_engine = FoodRankingEngine()
        self.activity_engine = ActivityEngine()
        print("✓ CDSS ready")

    def run(self, cgm_readings, carbs=0, protein=0, fat=0, top_k=10) -> Dict:
        cgm_readings = list(cgm_readings)
        if len(cgm_readings) < 10:
            cgm_readings = ([cgm_readings[0]] * (10 - len(cgm_readings))) + cgm_readings
        current_glucose = float(cgm_readings[-1])

        sequence = build_lstm_input(
            cgm_readings, carbs=carbs, protein=protein,
            fat=fat, window_size=self.config.WINDOW_SIZE
        )
        x = sequence[np.newaxis, :, :]

        mean_pred, lower, upper = self.prediction_engine.predict_glucose(x)

        risk_info = self.prediction_engine.compute_risk(
            current_glucose, mean_pred, lower, upper
        )
        risk_safe = safe_risk(risk_info)
        dominant  = risk_safe["dominant_risk"]

        food_recs = self.ranking_engine.rank(
            self.food_df, risk_safe["risk_level"], current_glucose,
            top_k=top_k, dominant_risk=dominant
        )
        meal_plan = self.ranking_engine.meal_plan(
            self.food_df, risk_safe["risk_level"], current_glucose,
            dominant_risk=dominant
        )
        activity = self.activity_engine.recommend(risk_info)

        return {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk":       risk_safe,
            "predictions": {
                "30min":  {
                    "mean":  round(float(mean_pred[0]), 1),
                    "lower": round(float(lower[0]), 1),
                    "upper": round(float(upper[0]), 1),
                },
                "60min":  {
                    "mean":  round(float(mean_pred[1]), 1),
                    "lower": round(float(lower[1]), 1),
                    "upper": round(float(upper[1]), 1),
                },
                "120min": {
                    "mean":  round(float(mean_pred[2]), 1),
                    "lower": round(float(lower[2]), 1),
                    "upper": round(float(upper[2]), 1),
                },
            },
            "food_recommendations": food_recs,
            "meal_plan":            meal_plan,
            "activity":             activity,
        }
