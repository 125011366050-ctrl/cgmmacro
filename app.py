import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import ClinicalOrchestrator, Config

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

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📥 Patient Input")

    cgm_input = st.text_area(
        "CGM Readings (comma-separated, mg/dL)",
        value="120, 125, 130, 140, 155, 165, 160, 158, 155, 150",
        help="Enter at least 10 readings, most recent last"
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

    # Historical trace
    hist_x = list(range(-len(cgm_readings) + 1, 1))
    hist_y = cgm_readings

    # Forecast points
    fut_x  = [0, 30, 60, 120]
    fut_y  = [current,
              preds["30min"]["mean"],
              preds["60min"]["mean"],
              preds["120min"]["mean"]]

    lower  = [current,
              preds["30min"]["lower"],
              preds["60min"]["lower"],
              preds["120min"]["lower"]]

    upper  = [current,
              preds["30min"]["upper"],
              preds["60min"]["upper"],
              preds["120min"]["upper"]]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        name="CGM History",
        line=dict(color="#6366f1", width=2),
        mode="lines+markers",
        marker=dict(size=4)
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(251,146,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        hoverinfo="skip"
    ))

    # Forecast mean
    fig.add_trace(go.Scatter(
        x=fut_x, y=fut_y,
        name="Forecast (mean)",
        line=dict(color="#f97316", width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond")
    ))

    # Clinical reference lines
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

    # ── ROW 1: key metrics ──────────────────────────────────────────────────
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

    # ── ROW 2: risk summary ─────────────────────────────────────────────────
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(
            f"### {risk_badge(risk['risk_level'])} &nbsp; "
            f"Dominant: **{DOMINANT_LABEL.get(risk['dominant_risk'], risk['dominant_risk'])}**",
            unsafe_allow_html=True
        )
        st.info(f"📋 {risk['clinical_summary']}")

        # Bidirectional risk bar
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("⬆️ Upward Spike",   f"{risk['upward_spike']:.1f} mg/dL")
            st.metric("Peak (worst-case)", f"{risk.get('worst_peak', risk['peak']):.1f} mg/dL")
        with col_b:
            st.metric("⬇️ Downward Drop",  f"{risk['downward_drop']:.1f} mg/dL")
            st.metric("Trough (worst-case)",f"{risk.get('worst_trough', risk['trough']):.1f} mg/dL")

        # Hypo flags
        if risk["hypo_risk"]:
            st.error("🚨 HYPOGLYCEMIA RISK — worst-case trough below 70 mg/dL")
        elif risk["hypo_warning"]:
            st.warning("⚠️ HYPO WARNING — worst-case trough below 90 mg/dL")

    with col_right:
        st.plotly_chart(
            build_risk_gauge(risk["risk_score"], risk["risk_level"]),
            use_container_width=True
        )

    # ── ROW 3: forecast chart ────────────────────────────────────────────────
    st.markdown("---")
    st.plotly_chart(
        build_forecast_chart(cgm_readings, result),
        use_container_width=True
    )

    # ── ROW 4: activity ──────────────────────────────────────────────────────
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

    # ── ROW 5: food recommendations ──────────────────────────────────────────
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

    # ── ROW 6: uncertainty details ───────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 Forecast Uncertainty Details"):
        unc = risk.get("uncertainty", {})
        u_lower = unc.get("lower", [])
        u_upper = unc.get("upper", [])
        u_std   = unc.get("std", [8, 12, 15])

        if u_lower and u_upper:
            import pandas as pd
            unc_df = pd.DataFrame({
                "Horizon":    ["30 min", "60 min", "120 min"],
                "Mean":       [preds["30min"]["mean"],
                               preds["60min"]["mean"],
                               preds["120min"]["mean"]],
                "Lower 95% CI": [round(v, 1) for v in u_lower],
                "Upper 95% CI": [round(v, 1) for v in u_upper],
                "Std (mg/dL)":  [round(v, 1) for v in u_std],
                "CI Width":     [round(u - l, 1) for l, u in zip(u_lower, u_upper)],
            })
            st.dataframe(unc_df, use_container_width=True, hide_index=True)
            st.caption(
                "Uncertainty grows with forecast horizon — consistent with CGM literature. "
                "Risk classification uses worst-case CI bounds, not just mean predictions."
            )

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"⏱️ Analysis timestamp: {result['timestamp']} &nbsp;|&nbsp; "
        f"Risk score: {risk['risk_score']} &nbsp;|&nbsp; "
        f"Drop severity: {risk['drop_severity']} &nbsp;|&nbsp; "
        f"Trend: {risk['trend_direction']}"
    )

import pandas as pd  # ensure available for meal plan section
