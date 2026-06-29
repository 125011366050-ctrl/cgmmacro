# ============================================================
# VERCEL-OPTIMIZED engine.py - Complete Single File
# Serves both API endpoints and beautiful frontend UI
# All files in project root - NO subfolders
# ============================================================

import numpy as np
import pandas as pd
import os
import warnings
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - All files in project root
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    DATA_PATH: str = BASE_DIR
    FEATURE_SCALER_FILE: str = os.path.join(BASE_DIR, "feature_scaler.pkl")
    GLUCOSE_SCALER_FILE: str = os.path.join(BASE_DIR, "glucose_scaler.pkl")
    LSTM_MODEL_FILE: str = os.path.join(BASE_DIR, "lstm_encoder_trained.pth")
    TABNET_MODEL_FILE: str = os.path.join(BASE_DIR, "tabnet_on_learned_embeddings.zip")
    EMBEDDING_SCALER_FILE: str = os.path.join(BASE_DIR, "embedding_scaler.pkl")
    FOOD_DB_FILE: str = os.path.join(BASE_DIR, "Indian_Foods_GI_GL_Database.xlsx")
    DEVICE: str = "cpu"
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
    SEVERE_HYPO_THRESHOLD: float = 55.0
    CRITICAL_DROP: float = 50.0
    MODERATE_DROP: float = 35.0
    DROP_HIGH_ALERT: float = 70.0
    DROP_MODERATE_ALERT: float = 50.0
    DROP_CAUTION: float = 35.0
    VELOCITY_HIGH_RISK: float = 3.0
    VELOCITY_MEDIUM_RISK: float = 1.5
    HYPERGLYCEMIA_THRESHOLD: float = 180.0
    RAPID_FALL_THRESHOLD: float = 30.0
    RAPID_RISE_THRESHOLD: float = 40.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
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


def safe_risk(risk: dict) -> dict:
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
        "trend_strength":   risk.get("trend_strength", 0.0),
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
    return df


# ============================================================
# FOOD RANKING ENGINE
# ============================================================
class FoodRankingEngine:
    def __init__(self):
        pass

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
             dominant_risk: str = "NONE",
             future_glucose: float = None,
             predictions: List[float] = None) -> pd.DataFrame:
        
        if df.empty:
            return pd.DataFrame()
        
        filtered = self.filter_by_risk(df, risk_level, dominant_risk)
        out = filtered.copy()
        
        if future_glucose is None:
            future_glucose = current_glucose
        
        spikes = np.array([
            self.estimate_spike(row.to_dict(), future_glucose)
            for _, row in out.iterrows()
        ], dtype=float)
        
        out["Predicted_Spike"] = spikes
        out["Predicted_Peak"] = spikes + future_glucose
        out["Current_Glucose"] = current_glucose
        
        out["Reason"] = np.where(
            out["Predicted_Spike"] > 40,
            "High glycemic impact - choose carefully",
            "Low glycemic impact - good choice"
        )
        
        if predictions is not None and len(predictions) >= 3:
            out["Pred_30min"] = predictions[0]
            out["Pred_60min"] = predictions[1]
            out["Pred_120min"] = predictions[2]
            
            avg_pred = np.mean(predictions)
            if avg_pred > 180:
                out["Reason"] = "Predicted high glucose - low GI recommended"
                out["Target_Score"] = 1.0 - (out["Predicted_Spike"] / 80.0)
            elif avg_pred < 100:
                out["Reason"] = "Predicted low glucose - higher GI acceptable"
                out["Target_Score"] = out["Predicted_Spike"] / 80.0
            else:
                out["Reason"] = "Balanced glucose - maintain healthy choices"
                out["Target_Score"] = 0.5
        else:
            out["Target_Score"] = 0.5
        
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
        
        if predictions is not None:
            w["target"] = 0.20
            total = sum(w.values())
            for key in w:
                w[key] = w[key] / (total + 0.20)
            
            out["Score"] = (
                w["spike"] * (1 - spike_n) +
                w["gi"]    * (1 - gi_n)    +
                w["gl"]    * (1 - gl_n)    +
                w["protein"] * prot_n      +
                w["fiber"]   * fiber_n     +
                w["target"] * out["Target_Score"]
            )
        else:
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
                  current_glucose: float, dominant_risk: str = "NONE",
                  future_glucose: float = None,
                  predictions: List[float] = None) -> Dict:
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
                             top_k=3, dominant_risk=dominant_risk,
                             future_glucose=future_glucose,
                             predictions=predictions)
            if not ranked.empty:
                plan[meal] = ranked[["Food_Name", "GI", "GL",
                                    "Predicted_Spike", "Score",
                                    "Reason", "Recommendation"]].to_dict("records")
            else:
                plan[meal] = []
        return plan


# ============================================================
# ACTIVITY ENGINE
# ============================================================
class ActivityEngine:
    def __init__(self):
        pass

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


# ============================================================
# VERCEL-OPTIMIZED PREDICTION ENGINE
# ============================================================
class PredictionEngine:
    """Lazy-loaded prediction engine optimized for serverless environments"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Config = None):
        if cls._instance is None:
            cls._instance = super(PredictionEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Config = None):
        if self._initialized:
            return
        
        if config is None:
            config = Config()
        
        self.config = config
        self.device = "cpu"
        self.rng = np.random.default_rng(seed=config.SEED)
        
        # Lazy-loaded attributes
        self._feature_scaler = None
        self._glucose_scaler = None
        self._scaler_mode = None
        self._lstm = None
        self._tabnet = None
        self._embedding_scaler = None
        self._models_loaded = False
        
        self._initialized = True
        print("✓ PredictionEngine initialized (models will load on first use)")

    def _load_models(self):
        """Lazy load models only when needed"""
        if self._models_loaded:
            return
        
        try:
            import torch
            import joblib
            import torch.nn as nn
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError as e:
            raise ImportError(f"Required ML packages not installed: {e}")
        
        # Load scalers from project root
        if not os.path.exists(self.config.FEATURE_SCALER_FILE):
            raise FileNotFoundError(f"Missing model file: {self.config.FEATURE_SCALER_FILE}")
        self._feature_scaler = joblib.load(self.config.FEATURE_SCALER_FILE)
        print("✓ Feature scaler loaded")
        
        if not os.path.exists(self.config.GLUCOSE_SCALER_FILE):
            raise FileNotFoundError(f"Missing model file: {self.config.GLUCOSE_SCALER_FILE}")
        self._glucose_scaler = joblib.load(self.config.GLUCOSE_SCALER_FILE)
        self._scaler_mode = self._probe_scaler(self._glucose_scaler, self.config.N_HORIZONS)
        print(f"✓ Glucose scaler loaded — mode: {self._scaler_mode}")
        
        # Define LSTM Encoder class
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
        
        # Load LSTM
        if not os.path.exists(self.config.LSTM_MODEL_FILE):
            raise FileNotFoundError(f"Missing model file: {self.config.LSTM_MODEL_FILE}")
        
        self._lstm = LSTMEncoder(
            input_size=self.config.INPUT_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            n_horizons=self.config.N_HORIZONS
        )
        self._lstm.load_state_dict(
            torch.load(self.config.LSTM_MODEL_FILE, map_location="cpu"),
            strict=False
        )
        self._lstm.eval()
        for param in self._lstm.parameters():
            param.requires_grad = False
        print("✓ LSTM loaded and frozen")
        
        # Load TabNet
        if not os.path.exists(self.config.TABNET_MODEL_FILE):
            raise FileNotFoundError(f"Missing model file: {self.config.TABNET_MODEL_FILE}")
        self._tabnet = TabNetRegressor()
        self._tabnet.load_model(self.config.TABNET_MODEL_FILE)
        print("✓ TabNet loaded")
        
        # Load embedding scaler (optional)
        if os.path.exists(self.config.EMBEDDING_SCALER_FILE):
            self._embedding_scaler = joblib.load(self.config.EMBEDDING_SCALER_FILE)
            print("✓ Embedding scaler loaded")
        else:
            self._embedding_scaler = None
            print("⚠️ No embedding scaler — using z-norm fallback")
        
        self._models_loaded = True
        print("✓ All models loaded successfully")

    def _probe_scaler(self, scaler, n_horizons=3):
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

    def _scale_features(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        x_scaled = self._feature_scaler.transform(x_flat)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return x_scaled.reshape(original_shape)

    def _inverse_scale(self, raw: np.ndarray) -> np.ndarray:
        raw = np.array(raw, dtype=np.float32).reshape(-1)
        try:
            if self._scaler_mode == 'multi':
                result = self._glucose_scaler.inverse_transform(raw.reshape(1, -1)).flatten()
            elif self._scaler_mode == 'single':
                result = np.array([
                    self._glucose_scaler.inverse_transform([[v]])[0][0] for v in raw
                ])
            else:
                result = raw.copy()
            result = np.nan_to_num(result, nan=120.0, posinf=400.0, neginf=40.0)
            return result
        except Exception as e:
            print(f"⚠️ inverse_scale failed: {e} — returning raw")
            return raw.copy()

    def predict_glucose(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        self._load_models()
        
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        current_glucose = float(x[0, -1, 0])

        x_scaled = self._scale_features(x)
        t = torch.from_numpy(x_scaled).float()
        with torch.no_grad():
            emb = self._lstm.get_embedding(t).cpu().numpy().astype(np.float32)
            emb = emb.reshape(emb.shape[0], -1)

            if self._embedding_scaler is not None:
                emb = self._embedding_scaler.transform(emb)
            else:
                emb_mean = np.mean(emb, axis=1, keepdims=True)
                emb_std = np.std(emb, axis=1, keepdims=True)
                emb = (emb - emb_mean) / (emb_std + 1e-6)
            emb = np.clip(emb, -5, 5)
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

            raw = self._tabnet.predict(emb)
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
                                hypo_warning: bool, trend_strength: float) -> str:
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
            if trend_strength > 1.5:
                return (f"HIGH GLUCOSE RISING: Current {current:.0f} mg/dL with rapid "
                        f"rise at {rate} mg/dL/min (trend strength {trend_strength:.1f}). "
                        f"Predicted spike of {upward_spike:.0f} mg/dL. "
                        f"Immediate glucose control intervention needed.")
            else:
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

        trend_strength = float(np.polyfit(range(len(predictions)), predictions, 1)[0])

        hypo_risk = np.mean(predictions < self.config.HYPO_THRESHOLD) >= 0.6
        hypo_warning = np.mean(predictions < self.config.HYPO_WARNING) >= 0.5

        hyper_risk = (
            (current >= self.config.WARNING_GLUCOSE and np.max(predictions) >= self.config.WARNING_GLUCOSE) or
            (trend_strength > 1.5 and current >= 150) or
            (np.mean(predictions >= self.config.WARNING_GLUCOSE) >= 0.5)
        )
        
        if trend_strength > 2.0 and current >= 140:
            hyper_risk = True

        velocity_high = abs(velocity) >= self.config.VELOCITY_HIGH_RISK and \
                        abs(predictions[0] - current) >= 20

        drop_risk_high   = downward_drop >= self.config.CRITICAL_DROP
        drop_risk_medium = downward_drop >= self.config.MODERATE_DROP
        spike_risk_high  = upward_spike >= self.config.MEDIUM_SPIKE
        spike_risk_medium = upward_spike >= self.config.LOW_SPIKE

        hypo_score = int(hypo_risk) * 3 + int(hypo_warning) * 2
        drop_score = int(drop_risk_high) * 3 + int(drop_risk_medium) * 2
        hyper_score = int(hyper_risk) * 3 + int(spike_risk_high) * 2 + int(spike_risk_medium) * 1
        velocity_score = int(velocity_high) * 2
        
        if trend_strength > 1.0:
            hyper_score += 1
        if trend_strength > 1.5:
            hyper_score += 1
        if trend_strength > 2.0:
            hyper_score += 1
        
        if current >= self.config.HYPERGLYCEMIA_THRESHOLD and trend_strength > 0.5:
            hyper_score = max(hyper_score, 10)
            trend_direction = "RISING"
        
        if current >= self.config.HYPERGLYCEMIA_THRESHOLD:
            drop_score = 0
        
        risk_scores = {
            "HYPOGLYCEMIA": hypo_score,
            "DROP_RISK": drop_score,
            "HYPERGLYCEMIA": hyper_score,
            "VELOCITY_RISK": velocity_score
        }
        
        if max(risk_scores.values()) >= 3:
            risk_level = "HIGH"
        elif max(risk_scores.values()) >= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        def is_consistently_rising(arr, window=3):
            if len(arr) < window:
                return False
            diffs = np.diff(arr[:window])
            return np.all(diffs > -2)

        def is_consistently_falling(arr, window=3):
            if len(arr) < window:
                return False
            diffs = np.diff(arr[:window])
            return np.all(diffs < 2)

        max_score = max(risk_scores.values())
        top_risks = [r for r, s in risk_scores.items() if s == max_score]
        
        if current >= self.config.HYPERGLYCEMIA_THRESHOLD:
            dominant_risk = "HYPERGLYCEMIA"
            trend_direction = "RISING"
            if velocity < 0:
                velocity = abs(velocity)
        elif max_score >= 3:
            if "HYPOGLYCEMIA" in top_risks and is_consistently_falling(predictions):
                dominant_risk = "HYPOGLYCEMIA"
            elif "HYPERGLYCEMIA" in top_risks and (is_consistently_rising(predictions) or trend_strength > 1.0):
                dominant_risk = "HYPERGLYCEMIA"
            elif "DROP_RISK" in top_risks:
                dominant_risk = "DROP_RISK"
            else:
                dominant_risk = top_risks[0]
        elif max_score >= 2:
            if "HYPERGLYCEMIA" in top_risks:
                dominant_risk = "HYPERGLYCEMIA"
            elif "HYPOGLYCEMIA" in top_risks:
                dominant_risk = "HYPOGLYCEMIA"
            elif "DROP_RISK" in top_risks:
                dominant_risk = "DROP_RISK"
            else:
                dominant_risk = top_risks[0]
        else:
            dominant_risk = "NONE"

        if current >= self.config.HYPERGLYCEMIA_THRESHOLD:
            if "DROP" in dominant_risk:
                dominant_risk = "HYPERGLYCEMIA"
            risk_level = "HIGH"

        risk_score = self._compute_risk_score(
            current, upward_spike, downward_drop, trough, velocity
        )
        clinical_summary = self._build_clinical_summary(
            risk_level, dominant_risk, current, trough,
            upward_spike, downward_drop, velocity, hypo_risk, 
            hypo_warning, trend_strength
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
            "trend_strength":   float(trend_strength),
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


# ============================================================
# CLINICAL ORCHESTRATOR
# ============================================================
class ClinicalOrchestrator:
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Config = None):
        if cls._instance is None:
            cls._instance = super(ClinicalOrchestrator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Config = None):
        if self._initialized:
            return
        
        if config is None:
            config = Config()
        self.config = config
        
        self._prediction_engine = PredictionEngine(config)
        
        if not os.path.exists(config.FOOD_DB_FILE):
            raise FileNotFoundError(f"Missing food database: {config.FOOD_DB_FILE}")
        self.food_df = load_food_database(config.FOOD_DB_FILE)
        
        self.ranking_engine = FoodRankingEngine()
        self.activity_engine = ActivityEngine()
        
        self._initialized = True
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

        mean_pred, lower, upper = self._prediction_engine.predict_glucose(x)

        risk_info = self._prediction_engine.compute_risk(
            current_glucose, mean_pred, lower, upper
        )
        risk_safe = safe_risk(risk_info)
        dominant  = risk_safe["dominant_risk"]
        
        pred_30 = mean_pred[0]
        pred_60 = mean_pred[1]
        pred_120 = mean_pred[2]
        
        future_glucose = 0.5 * pred_30 + 0.3 * pred_60 + 0.2 * pred_120
        
        predictions_list = [pred_30, pred_60, pred_120]

        food_recs = self.ranking_engine.rank(
            self.food_df, 
            risk_safe["risk_level"], 
            current_glucose,
            top_k=top_k, 
            dominant_risk=dominant,
            future_glucose=future_glucose,
            predictions=predictions_list
        )
        
        meal_plan = self.ranking_engine.meal_plan(
            self.food_df, 
            risk_safe["risk_level"], 
            current_glucose,
            dominant_risk=dominant,
            future_glucose=future_glucose,
            predictions=predictions_list
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
                "weighted_future": round(float(future_glucose), 1),
                "max_future": round(float(max(predictions_list)), 1),
                "min_future": round(float(min(predictions_list)), 1),
            },
            "food_recommendations": food_recs.to_dict("records") if not food_recs.empty else [],
            "meal_plan":            meal_plan,
            "activity":             activity,
        }


# ============================================================
# SINGLETON ACCESSOR
# ============================================================
_ENGINE = None

def get_engine() -> ClinicalOrchestrator:
    global _ENGINE
    if _ENGINE is None:
        config = Config()
        _ENGINE = ClinicalOrchestrator(config)
    return _ENGINE


# ============================================================
# FASTAPI APP WITH FRONTEND UI
# ============================================================
app = FastAPI(title="GlucoseGuard CDSS", description="Clinical Decision Support System")

class PredictionRequest(BaseModel):
    cgm_readings: List[float]
    carbs: Optional[float] = 0.0
    protein: Optional[float] = 0.0
    fat: Optional[float] = 0.0
    top_k: Optional[int] = 10

# ============================================================
# API ENDPOINTS
# ============================================================
@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        engine = get_engine()
        result = engine.run(
            cgm_readings=request.cgm_readings,
            carbs=request.carbs,
            protein=request.protein,
            fat=request.fat,
            top_k=request.top_k
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "healthy", "models_loaded": _ENGINE is not None}

# ============================================================
# FRONTEND UI - Beautiful HTML Interface
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GlucoseGuard CDSS</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e9ecef;
        }
        .card h3 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .slider-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .slider-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .slider-item label {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 2px;
        }
        .slider-item input[type="range"] {
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: #667eea;
            border-radius: 3px;
            outline: none;
        }
        .slider-item input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #764ba2;
            border-radius: 50%;
            cursor: pointer;
        }
        .slider-value {
            font-size: 0.9em;
            font-weight: bold;
            color: #333;
            margin-top: 2px;
        }
        .presets {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }
        .preset-btn {
            padding: 8px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9em;
        }
        .preset-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .meal-input {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        .meal-input input {
            padding: 8px 12px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            font-size: 1em;
            width: 100%;
        }
        .analyze-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        .analyze-btn:hover {
            transform: translateY(-2px);
        }
        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .results {
            display: none;
            margin-top: 30px;
            border-top: 2px solid #e9ecef;
            padding-top: 30px;
        }
        .results.show {
            display: block;
        }
        .risk-banner {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            font-size: 1.2em;
            font-weight: bold;
        }
        .risk-high { background: #dc3545; color: white; }
        .risk-medium { background: #ffc107; color: #000; }
        .risk-low { background: #28a745; color: white; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        @media (max-width: 768px) {
            .metrics { grid-template-columns: repeat(3, 1fr); }
        }
        .metric {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .metric .label { font-size: 0.8em; color: #666; }
        .metric .value { font-size: 1.3em; font-weight: bold; color: #333; }
        .metric .delta { font-size: 0.8em; }
        .tabs {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            border-bottom: 2px solid #e9ecef;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: bold;
        }
        .tab-content {
            display: none;
            padding: 20px 0;
        }
        .tab-content.active {
            display: block;
        }
        .table-wrap {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        th {
            background: #f1f3f5;
            font-weight: 600;
        }
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            display: none;
        }
        .error.show {
            display: block;
        }
        .summary-text {
            background: #d4edda;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 GlucoseGuard CDSS</h1>
        <div class="subtitle">Clinical Decision Support System for Diabetes Management</div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 CGM Readings (10 values)</h3>
                <div class="slider-group" id="sliders"></div>
                <div class="presets">
                    <button class="preset-btn" onclick="loadPreset('normal')">📊 Normal</button>
                    <button class="preset-btn" onclick="loadPreset('rising')">📈 Rising</button>
                    <button class="preset-btn" onclick="loadPreset('falling')">📉 Falling</button>
                    <button class="preset-btn" onclick="loadPreset('spike')">🎯 Spike</button>
                </div>
            </div>
            <div class="card">
                <h3>🍽️ Meal Context</h3>
                <div class="meal-input">
                    <div>
                        <label>Carbs (g)</label>
                        <input type="number" id="carbs" value="0" min="0" max="200">
                    </div>
                    <div>
                        <label>Protein (g)</label>
                        <input type="number" id="protein" value="0" min="0" max="100">
                    </div>
                    <div>
                        <label>Fat (g)</label>
                        <input type="number" id="fat" value="0" min="0" max="100">
                    </div>
                </div>
                <button class="analyze-btn" onclick="analyze()">🚀 Analyze</button>
            </div>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Analyzing glucose data...</p>
        </div>
        
        <div id="error" class="error"></div>
        
        <div id="results" class="results">
            <div id="riskBanner" class="risk-banner"></div>
            
            <div class="metrics" id="metrics"></div>
            
            <div class="summary-text" id="summary"></div>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('details')">📊 Details</div>
                <div class="tab" onclick="switchTab('food')">🍽️ Food</div>
                <div class="tab" onclick="switchTab('activity')">🏃 Activity</div>
            </div>
            
            <div id="tab-details" class="tab-content active">
                <div id="detailsContent"></div>
            </div>
            <div id="tab-food" class="tab-content">
                <div id="foodContent"></div>
            </div>
            <div id="tab-activity" class="tab-content">
                <div id="activityContent"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Preset data
        const PRESETS = {
            normal: [120, 122, 118, 121, 125, 128, 130, 135, 132, 129],
            rising: [120, 125, 130, 135, 140, 145, 150, 155, 160, 165],
            falling: [180, 175, 170, 165, 160, 155, 150, 145, 140, 135],
            spike: [120, 122, 118, 130, 165, 180, 175, 160, 145, 130]
        };
        
        let currentData = [...PRESETS.normal];
        
        // Initialize sliders
        function initSliders() {
            const container = document.getElementById('sliders');
            container.innerHTML = '';
            for (let i = 0; i < 10; i++) {
                const div = document.createElement('div');
                div.className = 'slider-item';
                div.innerHTML = `
                    <label>#${i+1}</label>
                    <input type="range" min="40" max="250" value="${currentData[i]}" 
                           oninput="updateSlider(${i}, this.value)">
                    <span class="slider-value" id="val-${i}">${currentData[i]}</span>
                `;
                container.appendChild(div);
            }
        }
        
        function updateSlider(idx, val) {
            currentData[idx] = parseInt(val);
            document.getElementById(`val-${idx}`).textContent = val;
        }
        
        function loadPreset(name) {
            const data = PRESETS[name];
            if (!data) return;
            currentData = [...data];
            for (let i = 0; i < 10; i++) {
                const slider = document.querySelector(`#sliders .slider-item:nth-child(${i+1}) input`);
                if (slider) {
                    slider.value = data[i];
                    document.getElementById(`val-${i}`).textContent = data[i];
                }
            }
        }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');
        }
        
        async function analyze() {
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const error = document.getElementById('error');
            
            loading.classList.add('show');
            results.classList.remove('show');
            error.classList.remove('show');
            
            const carbs = parseInt(document.getElementById('carbs').value) || 0;
            const protein = parseInt(document.getElementById('protein').value) || 0;
            const fat = parseInt(document.getElementById('fat').value) || 0;
            
            const payload = {
                cgm_readings: currentData,
                carbs: carbs,
                protein: protein,
                fat: fat,
                top_k: 10
            };
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const result = await response.json();
                
                if (!response.ok || result.status !== 'success') {
                    throw new Error(result.detail || 'Analysis failed');
                }
                
                displayResults(result.data);
                loading.classList.remove('show');
                results.classList.add('show');
                
            } catch (err) {
                loading.classList.remove('show');
                error.textContent = '❌ ' + err.message;
                error.classList.add('show');
            }
        }
        
        function displayResults(data) {
            const risk = data.risk || {};
            const preds = data.predictions || {};
            const food = data.food_recommendations || [];
            const mealPlan = data.meal_plan || {};
            const activity = data.activity || {};
            
            const riskLevel = risk.risk_level || 'LOW';
            const dominant = risk.dominant_risk || 'NONE';
            const summary = risk.clinical_summary || 'No summary';
            const current = risk.current || 0;
            const velocity = risk.glucose_velocity || 0;
            
            // Risk banner
            const banner = document.getElementById('riskBanner');
            banner.className = `risk-banner risk-${riskLevel.toLowerCase()}`;
            banner.innerHTML = `🚨 RISK: ${riskLevel} — ${dominant}<br><small>Trend: ${risk.trend_direction || 'STABLE'}</small>`;
            
            // Metrics
            const metricsHtml = `
                <div class="metric"><div class="label">Current</div><div class="value">${current.toFixed(0)} mg/dL</div></div>
                <div class="metric"><div class="label">30-min</div><div class="value">${(preds['30min']?.mean || 0).toFixed(0)}</div><div class="delta">${((preds['30min']?.mean || 0) - current).toFixed(0)}</div></div>
                <div class="metric"><div class="label">60-min</div><div class="value">${(preds['60min']?.mean || 0).toFixed(0)}</div><div class="delta">${((preds['60min']?.mean || 0) - current).toFixed(0)}</div></div>
                <div class="metric"><div class="label">120-min</div><div class="value">${(preds['120min']?.mean || 0).toFixed(0)}</div><div class="delta">${((preds['120min']?.mean || 0) - current).toFixed(0)}</div></div>
                <div class="metric"><div class="label">Velocity</div><div class="value">${velocity.toFixed(2)}</div></div>
            `;
            document.getElementById('metrics').innerHTML = metricsHtml;
            
            // Summary
            document.getElementById('summary').textContent = '🧠 ' + summary;
            
            // Details tab
            const detailsHtml = `
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                    <div>
                        <h4>Prediction Details</h4>
                        <table>
                            <tr><th>Horizon</th><th>Mean</th><th>Lower</th><th>Upper</th></tr>
                            <tr><td>30 min</td><td>${(preds['30min']?.mean || 0).toFixed(1)}</td><td>${(preds['30min']?.lower || 0).toFixed(1)}</td><td>${(preds['30min']?.upper || 0).toFixed(1)}</td></tr>
                            <tr><td>60 min</td><td>${(preds['60min']?.mean || 0).toFixed(1)}</td><td>${(preds['60min']?.lower || 0).toFixed(1)}</td><td>${(preds['60min']?.upper || 0).toFixed(1)}</td></tr>
                            <tr><td>120 min</td><td>${(preds['120min']?.mean || 0).toFixed(1)}</td><td>${(preds['120min']?.lower || 0).toFixed(1)}</td><td>${(preds['120min']?.upper || 0).toFixed(1)}</td></tr>
                        </table>
                    </div>
                    <div>
                        <h4>Risk Metrics</h4>
                        <table>
                            <tr><td>Peak</td><td>${(risk.peak || 0).toFixed(0)} mg/dL</td></tr>
                            <tr><td>Trough</td><td>${(risk.trough || 0).toFixed(0)} mg/dL</td></tr>
                            <tr><td>Risk Score</td><td>${(risk.risk_score || 0).toFixed(2)}</td></tr>
                            <tr><td>Requires Action</td><td>${risk.requires_action ? '✅ Yes' : '❌ No'}</td></tr>
                        </table>
                    </div>
                </div>
            `;
            document.getElementById('detailsContent').innerHTML = detailsHtml;
            
            // Food tab
            let foodHtml = '<h4>Top Food Recommendations</h4>';
            if (food && food.length > 0) {
                foodHtml += '<div class="table-wrap"><table><tr>';
                const headers = Object.keys(food[0]);
                headers.forEach(h => foodHtml += `<th>${h}</th>`);
                foodHtml += '</tr>';
                food.forEach(item => {
                    foodHtml += '<tr>';
                    headers.forEach(h => foodHtml += `<td>${item[h] || ''}</td>`);
                    foodHtml += '</tr>';
                });
                foodHtml += '</table></div>';
                
                if (Object.keys(mealPlan).length > 0) {
                    foodHtml += '<h4 style="margin-top:20px;">🥗 Meal Plan</h4>';
                    for (const [meal, items] of Object.entries(mealPlan)) {
                        if (items && items.length > 0) {
                            foodHtml += `<h5>${meal}</h5><div class="table-wrap"><table><tr>`;
                            const h = Object.keys(items[0]);
                            h.forEach(k => foodHtml += `<th>${k}</th>`);
                            foodHtml += '</tr>';
                            items.forEach(item => {
                                foodHtml += '<tr>';
                                h.forEach(k => foodHtml += `<td>${item[k] || ''}</td>`);
                                foodHtml += '</tr>';
                            });
                            foodHtml += '</table></div>';
                        }
                    }
                }
            } else {
                foodHtml += '<p>No food recommendations available</p>';
            }
            document.getElementById('foodContent').innerHTML = foodHtml;
            
            // Activity tab
            const activityHtml = `
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                    <div>
                        <p><strong>Activity:</strong> ${activity.activity || 'N/A'}</p>
                        <p><strong>Duration:</strong> ${activity.duration || 'N/A'}</p>
                        <p><strong>Timing:</strong> ${activity.timing || 'N/A'}</p>
                    </div>
                    <div>
                        <p><strong>Intensity:</strong> ${activity.intensity || 'N/A'}</p>
                        <p><strong>Calorie Burn:</strong> ${activity.calorie_burn || 'N/A'}</p>
                        <div style="background:#f8d7da;padding:10px;border-radius:6px;margin-top:10px;">
                            <strong>⚠️ Clinical Alert:</strong> ${activity.clinical_alert || 'No alert'}
                        </div>
                        <p style="font-size:0.9em;color:#666;margin-top:10px;"><em>${activity.evidence || 'No evidence'}</em></p>
                    </div>
                </div>
            `;
            document.getElementById('activityContent').innerHTML = activityHtml;
        }
        
        // Initialize
        initSliders();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI"""
    return HTML_TEMPLATE

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ============================================================
# MAIN (for local development)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GlucoseGuard CDSS - Starting Server")
    print("=" * 60)
    print("📊 API Endpoint: http://localhost:8000/api/predict")
    print("🌐 Frontend UI: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
