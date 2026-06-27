import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  NOTE: This file is fully self-contained. There is no engine.py.
#  The "ClinicalOrchestrator" below uses simple, transparent math
#  (linear trend + smoothing), NOT a trained ML model. It is a
#  placeholder so the app runs end-to-end. Replace `run()` with
#  real model inference when you have one.
# ─────────────────────────────────────────────────────────────


class Config:
    HYPO_THRESHOLD = 70
    HYPO_WARNING_THRESHOLD = 90
    HIGH_WARNING_THRESHOLD = 180
    CRITICAL_THRESHOLD = 200
    HORIZONS = [30, 60, 120]  # minutes
    SAMPLE_INTERVAL_MIN = 5   # assumed minutes between CGM readings


class ClinicalOrchestrator:
    def __init__(self, config: Config):
        self.cfg = config

    # ---------- forecasting ----------
    def _forecast(self, cgm_readings: list) -> dict:
        readings = np.array(cgm_readings, dtype=float)
        n = len(readings)
        current = float(readings[-1])

        # simple linear trend over last up-to-6 points
        window = readings[-6:] if n >= 2 else readings
        x = np.arange(len(window))
        if len(window) >= 2:
            slope, intercept = np.polyfit(x, window, 1)
        else:
            slope, intercept = 0.0, current

        preds = {}
        std_list = []
        lower_list = []
        upper_list = []
        for h in self.cfg.HORIZONS:
            steps = h / self.cfg.SAMPLE_INTERVAL_MIN
            mean_val = current + slope * steps
            mean_val = float(np.clip(mean_val, 40, 400))
            std = 5 + 0.07 * h  # uncertainty grows with horizon
            lower = float(np.clip(mean_val - 1.96 * std, 40, 400))
            upper = float(np.clip(mean_val + 1.96 * std, 40, 400))
            preds[f"{h}min"] = {
                "mean": round(mean_val, 1),
                "lower": round(lower, 1),
                "upper": round(upper, 1),
            }
            std_list.append(std)
            lower_list.append(lower)
            upper_list.append(upper)

        return {
            "preds": preds,
            "slope": slope,
            "std": std_list,
            "lower": lower_list,
            "upper": upper_list,
        }

    # ---------- risk ----------
    def _assess_risk(self, cgm_readings: list, forecast: dict) -> dict:
        readings = np.array(cgm_readings, dtype=float)
        current = float(readings[-1])
        preds = forecast["preds"]
        lower_list = forecast["lower"]
        upper_list = forecast["upper"]

        velocity = float((readings[-1] - readings[-2]) / self.cfg.SAMPLE_INTERVAL_MIN) if len(readings) >= 2 else 0.0

        worst_trough = min(lower_list + [current])
        worst_peak = max(upper_list + [current])
        peak = max([p["mean"] for p in preds.values()] + [current])
        trough = min([p["mean"] for p in preds.values()] + [current])

        hypo_risk = worst_trough < self.cfg.HYPO_THRESHOLD
        hypo_warning = (not hypo_risk) and (worst_trough < self.cfg.HYPO_WARNING_THRESHOLD)
        hyper_risk = worst_peak >= self.cfg.HIGH_WARNING_THRESHOLD

        upward_spike = max(0.0, peak - current)
        downward_drop = max(0.0, current - trough)

        if hypo_risk:
            dominant = "HYPOGLYCEMIA"
        elif hyper_risk and worst_peak >= self.cfg.CRITICAL_THRESHOLD:
            dominant = "HYPERGLYCEMIA"
        elif hypo_warning:
            dominant = "HYPO_WARNING"
        elif downward_drop > upward_spike and downward_drop > 15:
            dominant = "DROP_RISK"
        elif hyper_risk:
            dominant = "HYPERGLYCEMIA"
        else:
            dominant = "NONE"

        if dominant in ("HYPOGLYCEMIA", "HYPERGLYCEMIA"):
            risk_score = 0.85
            risk_level = "HIGH"
        elif dominant in ("HYPO_WARNING", "DROP_RISK"):
            risk_score = 0.55
            risk_level = "MEDIUM"
        else:
            risk_score = 0.2
            risk_level = "LOW"

        if velocity <= -2:
            velocity_risk = "Fast drop"
        elif velocity < 0:
            velocity_risk = "Slow drop"
        elif velocity == 0:
            velocity_risk = "Stable"
        elif velocity < 2:
            velocity_risk = "Slow rise"
        else:
            velocity_risk = "Fast rise"

        trend_direction = "Rising" if forecast["slope"] > 0.5 else ("Falling" if forecast["slope"] < -0.5 else "Stable")

        drop_severity = round(downward_drop / 10, 1)

        summaries = {
            "HYPOGLYCEMIA": "Worst-case forecast trough falls below the hypoglycemia threshold. Immediate carbohydrate intake recommended.",
            "HYPERGLYCEMIA": "Worst-case forecast peak exceeds the high glucose threshold. Monitor closely and consider correction per care plan.",
            "HYPO_WARNING": "Worst-case forecast trough is approaching the hypoglycemia threshold. Monitor closely.",
            "DROP_RISK": "Glucose is trending downward with a notable projected drop. Monitor and consider a small snack.",
            "NONE": "Glucose trend is within expected range. No immediate action indicated.",
        }

        return {
            "current": current,
            "glucose_velocity": velocity,
            "velocity_risk": velocity_risk,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "dominant_risk": dominant,
            "clinical_summary": summaries[dominant],
            "upward_spike": upward_spike,
            "downward_drop": downward_drop,
            "peak": peak,
            "trough": trough,
            "worst_peak": worst_peak,
            "worst_trough": worst_trough,
            "hypo_risk": hypo_risk,
            "hypo_warning": hypo_warning,
            "trend_direction": trend_direction,
            "drop_severity": drop_severity,
            "uncertainty": {
                "lower": [round(v, 1) for v in lower_list],
                "upper": [round(v, 1) for v in upper_list],
                "std": [round(v, 1) for v in forecast["std"]],
            },
        }

    # ---------- activity ----------
    def _recommend_activity(self, risk: dict) -> dict:
        if risk["hypo_risk"]:
            return {
                "activity": "Rest / avoid exertion",
                "duration": "N/A",
                "timing": "Now",
                "intensity": "None",
                "urgency_score": 0.9,
                "clinical_alert": "Treat hypoglycemia first (e.g. 15g fast-acting carbs) before any activity.",
                "evidence": "Exercise during hypoglycemia can worsen glucose drop and impair recovery.",
            }
        elif risk["dominant_risk"] == "HYPO_WARNING" or risk["dominant_risk"] == "DROP_RISK":
            return {
                "activity": "Light walk",
                "duration": "10-15 min",
                "timing": "After a small snack",
                "intensity": "Low",
                "urgency_score": 0.5,
                "clinical_alert": "Glucose is trending down — consider a small snack before activity.",
                "evidence": "Light activity with adequate fuel reduces hypoglycemia risk during exertion.",
            }
        elif risk["dominant_risk"] == "HYPERGLYCEMIA":
            return {
                "activity": "Moderate walk",
                "duration": "20-30 min",
                "timing": "Now",
                "intensity": "Moderate",
                "urgency_score": 0.6,
                "clinical_alert": "Moderate activity can help lower elevated glucose; avoid intense exercise if ketones are present.",
                "evidence": "Moderate aerobic activity improves insulin sensitivity and glucose uptake.",
            }
        else:
            return {
                "activity": "Normal activity",
                "duration": "As planned",
                "timing": "Anytime",
                "intensity": "Any",
                "urgency_score": 0.1,
                "clinical_alert": "",
                "evidence": "Glucose trend is within expected range.",
            }

    # ---------- food ----------
    def _recommend_food(self, carbs: float, protein: float, fat: float, top_k: int) -> pd.DataFrame:
        foods = [
            ("Greek Yogurt (plain)", 12, 3),
            ("Steel-cut Oats", 55, 13),
            ("Lentils", 32, 5),
            ("Apple", 36, 6),
            ("Almonds (small handful)", 15, 1),
            ("Grilled Chicken Breast", 0, 0),
            ("Quinoa", 53, 13),
            ("Broccoli", 15, 4),
            ("Sweet Potato", 63, 11),
            ("Chickpeas", 28, 8),
            ("Whole Wheat Bread", 71, 6),
            ("Brown Rice", 50, 10),
            ("Salmon", 0, 0),
            ("Carrots", 39, 5),
            ("Walnuts", 14, 1),
            ("Cottage Cheese", 3, 1),
            ("Banana", 51, 12),
            ("Hummus", 15, 4),
            ("Spinach Salad", 4, 1),
            ("Black Beans", 23, 5),
        ]
        rows = []
        for name, gi, gl in foods:
            spike = round(gl * 0.8 + carbs * 0.05 - protein * 0.05, 1)
            spike = max(0.0, spike)
            peak = round(120 + spike, 1)
            score = round(max(0, 100 - gl * 1.5 - spike), 1)
            rec = "Recommended" if score >= 60 else ("Caution" if score >= 35 else "Avoid")
            rows.append({
                "Food_Name": name, "GI": gi, "GL": gl,
                "Predicted_Spike": spike, "Predicted_Peak": peak,
                "Score": score, "Recommendation": rec,
            })
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df.head(top_k)

    def _build_meal_plan(self, food_df: pd.DataFrame) -> dict:
        top = food_df.to_dict("records")
        return {
            "Breakfast": top[0:2],
            "Lunch": top[2:4],
            "Dinner": top[4:6],
            "Snacks": top[6:8],
        }

    # ---------- entry point ----------
    def run(self, cgm_readings: list, carbs: float, protein: float, fat: float, top_k: int) -> dict:
        forecast = self._forecast(cgm_readings)
        risk = self._assess_risk(cgm_readings, forecast)
        activity = self._recommend_activity(risk)
        food_df = self._recommend_food(carbs, protein, fat, top_k)
        meal_plan = self._build_meal_plan(food_df)

        return {
            "predictions": forecast["preds"],
            "risk": risk,
            "activity": activity,
            "food_recommendations": food_df,
            "meal_plan": meal_plan,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# ─────────────────────────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_system():
    return ClinicalOrchestrator(Config())

orchestrator = load_system()

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CDSS — Glucose Monitor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Clinical Decision Support System")
st.caption("CGM-based glucose forecasting with bidirectional risk engine")
st.warning(
    "⚠️ Demo placeholder engine: forecasts use simple linear-trend math, not a trained "
    "clinical model. Do not use for real medical decisions.",
    icon="⚠️"
)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📥 Patient Input")

    cgm_input = st.text_area(
        "CGM Readings (comma-separated, mg/dL)",
        value="120, 125, 130, 140, 155, 165, 160, 158, 155, 150",
        help="Enter at least 3 readings, most recent last"
    )

    st.subheader("🍽️ Last Meal Macros")
    carbs   = st.slider("Carbohydrates (g)", 0, 150, 60)
    protein = st.slider("Protein (g)",        0, 100, 20)
    fat     = st.slider("Fat (g)",            0, 100, 15)
    top_k   = st.slider("Food recommendations (top N)", 5, 20, 10)

    run_btn = st.button("🔍 Analyse", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

RISK_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
RISK_ICON  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

DOMINANT_LABEL = {
    "HYPOGLYCEMIA": "🚨 Hypoglycemia",
    "DROP_RISK":    "⬇️ Drop Risk",
    "HYPERGLYCEMIA":"⬆️ Hyperglycemia",
    "HYPO_WARNING": "⚠️ Hypo Warning",
    "NONE":         "✅ None",
}

def risk_badge(level: str) -> str:
    color = RISK_COLOR.get(level, "#6b7280")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-weight:600;">{RISK_ICON.get(level,"")} {level}</span>'

def fmt_pred(p: dict) -> str:
    return f"{p['mean']} ({p['lower']}–{p['upper']}) mg/dL"

# ─────────────────────────────────────────────────────────────
#  GLUCOSE FORECAST CHART
# ─────────────────────────────────────────────────────────────

def build_forecast_chart(cgm_readings: list, result: dict) -> go.Figure:
    preds = result["predictions"]
    current = result["risk"]["current"]

    hist_x = list(range(-len(cgm_readings) + 1, 1))
    hist_y = cgm_readings

    fut_x  = [0, 30, 60, 120]
    fut_y  = [current, preds["30min"]["mean"], preds["60min"]["mean"], preds["120min"]["mean"]]
    lower  = [current, preds["30min"]["lower"], preds["60min"]["lower"], preds["120min"]["lower"]]
    upper  = [current, preds["30min"]["upper"], preds["60min"]["upper"], preds["120min"]["upper"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, name="CGM History",
        line=dict(color="#6366f1", width=2), mode="lines+markers", marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(251,146,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=fut_x, y=fut_y, name="Forecast (mean)",
        line=dict(color="#f97316", width=2, dash="dash"),
        mode="lines+markers", marker=dict(size=8, symbol="diamond")
    ))

    fig.add_hline(y=70,  line_dash="dot", line_color="#ef4444",
                  annotation_text="Hypo threshold (70)", annotation_position="right")
    fig.add_hline(y=180, line_dash="dot", line_color="#f59e0b",
                  annotation_text="High warning (180)",  annotation_position="right")
    fig.add_hline(y=200, line_dash="dot", line_color="#dc2626",
                  annotation_text="Critical (200)",      annotation_position="right")

    fig.update_layout(
        title="Glucose Forecast with 95% Confidence Interval",
        xaxis_title="Minutes (0 = now)",
        yaxis_title="Glucose (mg/dL)",
        yaxis=dict(range=[40, 420]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=10, r=80, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(100,100,100,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(100,100,100,0.1)")
    return fig

# ─────────────────────────────────────────────────────────────
#  RISK GAUGE CHART
# ─────────────────────────────────────────────────────────────

def build_risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    color = RISK_COLOR.get(risk_level, "#6b7280")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score * 100, 1),
        title={"text": f"Risk Score — {risk_level}", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100],"color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": risk_score * 100
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ─────────────────────────────────────────────────────────────
#  MAIN — RESULTS
# ─────────────────────────────────────────────────────────────

if run_btn:
    try:
        cgm_readings = [float(v.strip()) for v in cgm_input.split(",") if v.strip()]
        if len(cgm_readings) < 3:
            st.error("Please enter at least 3 CGM readings.")
            st.stop()
    except ValueError:
        st.error("Invalid CGM input — use comma-separated numbers.")
        st.stop()

    with st.spinner("Running clinical analysis..."):
        result = orchestrator.run(
            cgm_readings, carbs=carbs, protein=protein, fat=fat, top_k=top_k
        )

    risk  = result["risk"]
    preds = result["predictions"]
    act   = result["activity"]

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Current Glucose",   f"{risk['current']:.0f} mg/dL")
    c2.metric("30-min Forecast",   fmt_pred(preds["30min"]))
    c3.metric("60-min Forecast",   fmt_pred(preds["60min"]))
    c4.metric("120-min Forecast",  fmt_pred(preds["120min"]))
    c5.metric("Velocity",
              f"{risk['glucose_velocity']:+.2f} mg/dL/min",
              delta=risk["velocity_risk"],
              delta_color="inverse" if risk["glucose_velocity"] < 0 else "normal")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(
            f"### {risk_badge(risk['risk_level'])} &nbsp; "
            f"Dominant: **{DOMINANT_LABEL.get(risk['dominant_risk'], risk['dominant_risk'])}**",
            unsafe_allow_html=True
        )
        st.info(f"📋 {risk['clinical_summary']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("⬆️ Upward Spike",   f"{risk['upward_spike']:.1f} mg/dL")
            st.metric("Peak (worst-case)", f"{risk.get('worst_peak', risk['peak']):.1f} mg/dL")
        with col_b:
            st.metric("⬇️ Downward Drop",  f"{risk['downward_drop']:.1f} mg/dL")
            st.metric("Trough (worst-case)",f"{risk.get('worst_trough', risk['trough']):.1f} mg/dL")

        if risk["hypo_risk"]:
            st.error("🚨 HYPOGLYCEMIA RISK — worst-case trough below 70 mg/dL")
        elif risk["hypo_warning"]:
            st.warning("⚠️ HYPO WARNING — worst-case trough below 90 mg/dL")

    with col_right:
        st.plotly_chart(
            build_risk_gauge(risk["risk_score"], risk["risk_level"]),
            use_container_width=True
        )

    st.markdown("---")
    st.plotly_chart(
        build_forecast_chart(cgm_readings, result),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🏃 Activity Recommendation")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Activity",   act["activity"])
    a2.metric("Duration",   act["duration"])
    a3.metric("Timing",     act["timing"])
    a4.metric("Intensity",  act["intensity"])

    urgency = act["urgency_score"]
    st.progress(urgency, text=f"Urgency: {urgency:.0%}")
    if act["clinical_alert"]:
        if urgency >= 0.8:
            st.error(act["clinical_alert"])
        elif urgency >= 0.5:
            st.warning(act["clinical_alert"])
        else:
            st.info(act["clinical_alert"])

    st.caption(f"Evidence: {act['evidence']}")

    st.markdown("---")
    st.subheader("🍽️ Food Recommendations")

    tab1, tab2 = st.tabs(["Top Picks", "Meal Plan"])

    with tab1:
        food_df = result["food_recommendations"]
        if not food_df.empty:
            display_cols = ["Rank", "Food_Name", "GI", "GL",
                            "Predicted_Spike", "Predicted_Peak",
                            "Score", "Recommendation"]
            show_cols = [c for c in display_cols if c in food_df.columns]
            st.dataframe(
                food_df[show_cols].style.background_gradient(
                    subset=["Score"], cmap="RdYlGn"
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No food recommendations available.")

    with tab2:
        meal_plan = result["meal_plan"]
        mp_cols = st.columns(2)
        meals = list(meal_plan.items())
        for i, (meal, items) in enumerate(meals):
            with mp_cols[i % 2]:
                st.markdown(f"**{meal}**")
                if items:
                    mp_df = pd.DataFrame(items)
                    st.dataframe(mp_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No items found.")

    st.markdown("---")
    with st.expander("📊 Forecast Uncertainty Details"):
        unc = risk.get("uncertainty", {})
        u_lower = unc.get("lower", [])
        u_upper = unc.get("upper", [])
        u_std   = unc.get("std", [8, 12, 15])

        if u_lower and u_upper:
            unc_df = pd.DataFrame({
                "Horizon":    ["30 min", "60 min", "120 min"],
                "Mean":       [preds["30min"]["mean"], preds["60min"]["mean"], preds["120min"]["mean"]],
                "Lower 95% CI": [round(v, 1) for v in u_lower],
                "Upper 95% CI": [round(v, 1) for v in u_upper],
                "Std (mg/dL)":  [round(v, 1) for v in u_std],
                "CI Width":     [round(u - l, 1) for l, u in zip(u_lower, u_upper)],
            })
            st.dataframe(unc_df, use_container_width=True, hide_index=True)
            st.caption(
                "Uncertainty grows with forecast horizon. Risk classification uses "
                "worst-case CI bounds, not just mean predictions."
            )

    st.markdown("---")
    st.caption(
        f"⏱️ Analysis timestamp: {result['timestamp']} &nbsp;|&nbsp; "
        f"Risk score: {risk['risk_score']} &nbsp;|&nbsp; "
        f"Drop severity: {risk['drop_severity']} &nbsp;|&nbsp; "
        f"Trend: {risk['trend_direction']}"
    )
