import numpy as np
import pandas as pd
from typing import Dict


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
