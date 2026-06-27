import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import warnings
from datetime import datetime
from typing import Dict
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
    MAX_DROP_PER_STEP: float = 30.0
    MAX_RISE_PER_STEP: float = 60.0

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

class PredictionEngine:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.load_models()

    def load_models(self):
        base = self.config.DATA_PATH

        feature_scaler_path = os.path.join(base, "feature_scaler.pkl")
        if not os.path.exists(feature_scaler_path):
            raise FileNotFoundError(f"Feature scaler missing: {feature_scaler_path}")
        self.feature_scaler = joblib.load(feature_scaler_path)
        print("✓ Feature scaler loaded")

        scaler_path = os.path.join(base, "glucose_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        self.scaler_mode = _probe_scaler(self.scaler, self.config.N_HORIZONS)
        print(f"✓ Glucose scaler loaded — mode: {self.scaler_mode}")

        self.lstm = LSTMEncoder(
            input_size=self.config.INPUT_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            n_horizons=self.config.N_HORIZONS
        ).to(self.device)

        lstm_path = os.path.join(base, "lstm_encoder_trained.pth")
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Missing LSTM: {lstm_path}")
        self.lstm.load_state_dict(
            torch.load(lstm_path, map_location=self.device),
            strict=False
        )
        self.lstm.eval()
        for param in self.lstm.parameters():
            param.requires_grad = False
        print("✓ LSTM loaded and frozen")

        tabnet_path = os.path.join(base, "tabnet_on_learned_embeddings.zip")
        if not os.path.exists(tabnet_path):
            raise FileNotFoundError(f"Missing TabNet: {tabnet_path}")
        self.tabnet = TabNetRegressor()
        self.tabnet.load_model(tabnet_path)
        print("✓ TabNet loaded")

        embedding_scaler_path = os.path.join(base, "embedding_scaler.pkl")
        if os.path.exists(embedding_scaler_path):
            self.embedding_scaler = joblib.load(embedding_scaler_path)
            print("✓ Embedding scaler loaded")
        else:
            self.embedding_scaler = None
            print("⚠️ No embedding scaler — using z-norm fallback")

    def _scale_features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_scaler is None:
            raise ValueError("Feature scaler missing!")
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

    def _apply_physiology_clamp(self, pred: np.ndarray, current: float) -> np.ndarray:
        """
        Enforce physiological continuity.
        Max natural drop: ~30 mg/dL per 30-min horizon
        Max natural rise: ~60 mg/dL per 30-min horizon (post-meal)
        """
        max_drop = self.config.MAX_DROP_PER_STEP
        max_rise = self.config.MAX_RISE_PER_STEP
        pred = pred.copy()

        pred[0] = np.clip(pred[0], current - max_drop, current + max_rise)
        pred[1] = np.clip(pred[1], pred[0] - max_drop, pred[0] + max_rise)
        pred[2] = np.clip(pred[2], pred[1] - max_drop, pred[1] + max_rise)

        if np.any(np.diff(pred) < -40):
            print("⚠️ Physiology violation detected — applying smoothing")
            pred[1] = (pred[0] + pred[2]) / 2.0

        return pred

    def predict_glucose(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]
        x = x.astype(np.float32)

        # Extract current glucose from raw (unscaled) input before scaling
        current_glucose = float(x[0, -1, 0])

        x_scaled = self._scale_features(x)
        t = torch.from_numpy(x_scaled).float().to(self.device)

        with torch.no_grad():
            emb = self.lstm.get_embedding(t).cpu().numpy().astype(np.float32)

        emb = emb.reshape(emb.shape[0], -1)
        print(f"[DEBUG] EMB shape: {emb.shape} | mean: {emb.mean():.4f} | std: {emb.std():.4f}")

        if self.embedding_scaler is not None:
            emb = self.embedding_scaler.transform(emb)
        else:
            # z-norm per feature, not global — more stable
            emb_mean = np.mean(emb, axis=1, keepdims=True)
            emb_std = np.std(emb, axis=1, keepdims=True)
            emb = (emb - emb_mean) / (emb_std + 1e-6)

        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        raw = self.tabnet.predict(emb)
        raw = np.array(raw, dtype=np.float32).flatten()
        print(f"[DEBUG] RAW TabNet output: {raw}")

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        pred = self._inverse_scale(raw)
        print(f"[DEBUG] After inverse_scale: {pred}")

        pred = np.clip(pred, 40, 400)

        if len(pred) == 1:
            base_val = float(pred[0])
            pred = np.array([base_val, base_val, base_val], dtype=np.float32)

        # Apply physiological clamp using ORIGINAL current glucose
        pred = self._apply_physiology_clamp(pred, current_glucose)
        print(f"[DEBUG] After physiology clamp: {pred} | current was: {current_glucose:.1f}")

        return pred

    def compute_risk(self, current: float, predictions: np.ndarray) -> Dict:
        peak = float(np.max(predictions))
        trough = float(np.min(predictions))

        # Use signed spike — both directions matter
        raw_spike = float(peak - current)
        spike = abs(raw_spike)
        trend = float(predictions[0] - current)

        # Hypoglycemia check
        hypo_risk = trough < 70.0

        if current >= self.config.CRITICAL_GLUCOSE or spike >= self.config.MEDIUM_SPIKE or hypo_risk:
            risk_level = "HIGH"
        elif current >= self.config.WARNING_GLUCOSE or spike >= self.config.LOW_SPIKE:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "spike": spike,
            "raw_spike": raw_spike,
            "trend": trend,
            "peak": peak,
            "trough": trough,
            "current": float(current),
            "hypo_risk": hypo_risk,
            "predictions": predictions.tolist(),
            "requires_action": risk_level in ["MEDIUM", "HIGH"]
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
    print(f"✓ Food DB ready | GI range: {df['GI'].min():.0f}–{df['GI'].max():.0f}")
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

    def filter_by_risk(self, df: pd.DataFrame, risk_level: str) -> pd.DataFrame:
        if risk_level == "HIGH":
            filtered = df[(df["GI"] <= 40) & (df["GL"] <= 15)]
            if len(filtered) < 5:
                filtered = df[(df["GI"] <= 55) & (df["GL"] <= 20)]
        elif risk_level == "MEDIUM":
            filtered = df[(df["GI"] <= 55) & (df["GL"] <= 20)]
            if len(filtered) < 5:
                filtered = df[df["GI"] <= 70]
        else:
            filtered = df.copy()
        if len(filtered) < 5:
            filtered = df.nsmallest(max(10, len(df) // 2), "GI")
        return filtered.reset_index(drop=True)

    def rank(self, df: pd.DataFrame, risk_level: str,
             current_glucose: float, top_k: int = 10) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        filtered = self.filter_by_risk(df, risk_level)
        out = filtered.copy()
        spikes = np.array([
            self.estimate_spike(row.to_dict(), current_glucose)
            for _, row in out.iterrows()
        ], dtype=float)
        out["Predicted_Spike"] = spikes
        out["Predicted_Peak"] = spikes + current_glucose
        spike_n = self._normalize(spikes)
        gi_n = self._normalize(out["GI"].values)
        gl_n = self._normalize(out["GL"].values)
        prot_n = self._normalize(out["Protein"].values)
        fiber_n = self._normalize(out["Fiber"].values)
        if risk_level == "HIGH":
            w = dict(spike=0.35, gi=0.25, gl=0.20, protein=0.10, fiber=0.10)
        elif risk_level == "MEDIUM":
            w = dict(spike=0.25, gi=0.25, gl=0.20, protein=0.15, fiber=0.15)
        else:
            w = dict(spike=0.15, gi=0.20, gl=0.15, protein=0.25, fiber=0.25)
        out["Score"] = (
            w["spike"] * (1 - spike_n) +
            w["gi"] * (1 - gi_n) +
            w["gl"] * (1 - gl_n) +
            w["protein"] * prot_n +
            w["fiber"] * fiber_n
        )
        out = out.sort_values("Score", ascending=False).head(top_k).reset_index(drop=True)
        out["Rank"] = range(1, len(out) + 1)
        out["Recommendation"] = out["Score"].apply(
            lambda s: "⭐ Top Pick" if s > 0.75
            else "👍 Good Choice" if s > 0.55
            else "✓ Acceptable"
        )
        return out

    def meal_plan(self, df: pd.DataFrame, risk_level: str, current_glucose: float) -> Dict:
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
                subset = df.sample(min(30, len(df)), random_state=42)
            ranked = self.rank(subset, risk_level, current_glucose, top_k=3)
            if not ranked.empty:
                plan[meal] = ranked[["Food_Name", "GI", "GL", "Predicted_Spike", "Score"]].to_dict("records")
            else:
                plan[meal] = []
        return plan

class ActivityEngine:
    def recommend(self, risk_info: Dict) -> Dict:
        risk = risk_info["risk_level"]
        glucose = risk_info["current"]
        spike = risk_info["spike"]
        hypo = risk_info.get("hypo_risk", False)

        if hypo:
            return {
                "activity": "REST — Do NOT exercise",
                "duration": "Until glucose > 90 mg/dL",
                "timing": "IMMEDIATELY check glucose",
                "intensity": "None",
                "calorie_burn": "0 kcal",
                "clinical_alert": "🚨 HYPOGLYCEMIA RISK — Consume fast-acting carbs now",
                "evidence": "Exercise contraindicated below 70 mg/dL",
                "urgency_score": 1.0
            }
        elif glucose >= 200 or spike > 60:
            return {
                "activity": "URGENT — Brisk Walking",
                "duration": "40-45 minutes",
                "timing": "IMMEDIATELY within 10 minutes",
                "intensity": "High",
                "calorie_burn": "200-250 kcal",
                "clinical_alert": "🚨 CRITICAL — Do not remain sedentary",
                "evidence": "Emergency glucose reduction protocol",
                "urgency_score": 1.0
            }
        elif risk == "HIGH":
            return {
                "activity": "Brisk Walking / Aerobic Exercise",
                "duration": "30-45 minutes",
                "timing": "Within 15 min after meals",
                "intensity": "Moderate to High",
                "calorie_burn": "180-250 kcal",
                "clinical_alert": "⚠️ High risk — Activity essential",
                "evidence": "Reduces post-meal spike by 30-40%",
                "urgency_score": 0.8
            }
        elif risk == "MEDIUM":
            return {
                "activity": "Brisk Walking / Cycling",
                "duration": "20-30 minutes",
                "timing": "30-45 min after meals",
                "intensity": "Moderate",
                "calorie_burn": "120-180 kcal",
                "clinical_alert": "Activity recommended",
                "evidence": "Reduces post-meal glucose by 20-30%",
                "urgency_score": 0.5
            }
        else:
            return {
                "activity": "Light Walking / Yoga",
                "duration": "10-15 minutes",
                "timing": "Any time",
                "intensity": "Low",
                "calorie_burn": "50-80 kcal",
                "clinical_alert": "Maintain regular activity",
                "evidence": "Maintains baseline insulin sensitivity",
                "urgency_score": 0.2
            }

class ClinicalOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.prediction_engine = PredictionEngine(config)
        self.food_df = load_food_database(config.FOOD_FILE)
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

        predictions = self.prediction_engine.predict_glucose(x)
        risk_info = self.prediction_engine.compute_risk(current_glucose, predictions)
        food_recs = self.ranking_engine.rank(
            self.food_df, risk_info["risk_level"], current_glucose, top_k=top_k
        )
        meal_plan = self.ranking_engine.meal_plan(
            self.food_df, risk_info["risk_level"], current_glucose
        )
        activity = self.activity_engine.recommend(risk_info)

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk": risk_info,
            "predictions": {
                "30min": float(predictions[0]),
                "60min": float(predictions[1]),
                "120min": float(predictions[2])
            },
            "food_recommendations": food_recs,
            "meal_plan": meal_plan,
            "activity": activity
        }
