import streamlit as st
import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from reco import ClinicalOrchestrator, Config


@st.cache_resource
def load_system():
    config = Config()
    config.DATA_PATH = BASE_DIR
    config.INPUT_SIZE = 18
    config.FOOD_FILE = os.path.join(BASE_DIR, "Indian_Foods_GI_GL_Database (1).xlsx")
    return ClinicalOrchestrator(config)


st.set_page_config(page_title="CDSS — Glucose & Nutrition", layout="wide")
st.title("🩺 Clinical Decision Support System")
st.caption("CGM-Based Glucose Prediction + Personalised Food & Activity Recommendations")

try:
    system = load_system()
    st.success("✅ System loaded successfully")
except Exception as e:
    st.error(f"System failed to load: {e}")
    st.exception(e)
    st.stop()

st.markdown("---")
st.subheader("📋 Patient Input")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Meal Information**")
    meal_carbs = st.number_input("Carbohydrates (g)", 0, 300, 60, 5)
    meal_protein = st.number_input("Protein (g)", 0, 200, 20, 5)
    meal_fat = st.number_input("Fat (g)", 0, 200, 10, 5)

with col_b:
    st.markdown("**Last 10 CGM Readings — oldest → newest (mg/dL)**")
    defaults = [105, 108, 110, 112, 115, 117, 118, 119, 120, 120]
    cgm_inputs = []
    row1 = st.columns(5)
    row2 = st.columns(5)
    for i in range(10):
        col = row1[i] if i < 5 else row2[i - 5]
        with col:
            v = st.number_input(f"t-{9 - i}", 50, 400, defaults[i], 1, key=f"cgm_{i}")
            cgm_inputs.append(v)

st.markdown("---")

# debug mode toggle
debug_mode = st.checkbox("🔧 Debug Mode", value=False)

if st.button("🔍 Run Full Analysis", use_container_width=True):
    with st.spinner("Running prediction, food ranking, and activity plan..."):
        try:
            cgm_array = np.array(cgm_inputs, dtype=np.float32)

            # validate input range
            if np.any(cgm_array < 50) or np.any(cgm_array > 400):
                st.error("CGM values must be between 50 and 400 mg/dL")
                st.stop()

            if debug_mode:
                st.markdown("### 🔧 Debug — Raw Input")
                st.write("CGM array:", cgm_array)
                st.write("Shape:", cgm_array.shape)
                st.write("Carbs / Protein / Fat:", meal_carbs, meal_protein, meal_fat)

            result = system.run(
                cgm_readings=cgm_array,
                carbs=meal_carbs,
                protein=meal_protein,
                fat=meal_fat,
                top_k=10
            )

            current = float(cgm_inputs[-1])
            risk = result["risk"]
            preds = result["predictions"]
            risk_level = risk["risk_level"]
            spike = risk["spike"]
            peak = risk["peak"]
            trend = risk["trend"]
            activity = result["activity"]
            food_recs = result["food_recommendations"]
            meal_plan = result["meal_plan"]

            if debug_mode:
                st.markdown("### 🔧 Debug — Model Output")
                st.write("Raw predictions:", preds)
                st.write("Risk info:", risk)

            # ── GLUCOSE PREDICTIONS ──────────────────────────────
            st.markdown("---")
            st.subheader("📊 Predicted Glucose Levels")
            c1, c2, c3 = st.columns(3)
            c1.metric("30 min", f"{preds['30min']:.1f} mg/dL", delta=f"{preds['30min'] - current:+.1f}")
            c2.metric("60 min", f"{preds['60min']:.1f} mg/dL", delta=f"{preds['60min'] - current:+.1f}")
            c3.metric("90 min", f"{preds['90min']:.1f} mg/dL", delta=f"{preds['90min'] - current:+.1f}")

            # ── RISK ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("⚠️ Risk Assessment")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Current Glucose", f"{current:.0f} mg/dL")
            r2.metric("Predicted Peak", f"{peak:.0f} mg/dL")
            r3.metric("Glucose Spike", f"+{spike:.1f} mg/dL")
            r4.metric("30-min Trend", f"{trend:+.1f} mg/dL")

            if risk_level == "HIGH":
                st.error("🔴 HIGH RISK — Immediate action required")
                st.warning("⚡ Consider corrective insulin or immediate clinical review.")
            elif risk_level == "MEDIUM":
                st.warning("🟡 MEDIUM RISK — Monitor closely")
                st.info("📋 Avoid additional carbohydrates. Light activity recommended.")
            else:
                st.success("🟢 LOW RISK — Glucose trajectory stable")
                st.info("✅ Continue normal routine. Maintain regular activity.")

            # ── CGM SIGNAL SUMMARY ───────────────────────────────
            st.markdown("---")
            st.subheader("🔬 CGM Signal Summary")
            g = np.array(cgm_inputs, dtype=np.float32)
            cgm_slope = float(np.polyfit(np.arange(10), g, 1)[0])
            cgm_roc = float((g[-1] - g[-5]) / 5)
            cgm_cv = float(np.std(g) / (np.mean(g) + 1e-6) * 100)
            cgm_mean = float(np.mean(g))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean Glucose", f"{cgm_mean:.1f} mg/dL")
            m2.metric("Trend Slope", f"{cgm_slope:+.2f} mg/dL/step")
            m3.metric("Rate of Change", f"{cgm_roc:+.2f} mg/dL/step")
            m4.metric("Variability (CV)", f"{cgm_cv:.1f}%")

            # ── CGM CHART ────────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 CGM Trend")
            chart_data = pd.DataFrame({
                "Reading": list(range(1, 11)),
                "Glucose (mg/dL)": cgm_inputs
            }).set_index("Reading")
            st.line_chart(chart_data)

            # ── FOOD RECOMMENDATIONS ─────────────────────────────
            st.markdown("---")
            st.subheader("🍛 Food Recommendations")
            st.caption(
                f"Ranked by safety for **{risk_level}** risk — "
                f"scored on GI, GL, estimated spike, protein & fiber."
            )

            if isinstance(food_recs, pd.DataFrame) and not food_recs.empty:
                display_cols = [
                    "Rank", "Food_Name", "GI", "GL", "Carbs",
                    "Protein", "Fiber", "Predicted_Spike",
                    "Predicted_Peak", "Score", "Recommendation"
                ]
                show_cols = [c for c in display_cols if c in food_recs.columns]

                def highlight_spike(row):
                    val = row.get("Predicted_Spike", 0)
                    color = "#d4edda" if val < 20 else "#fff3cd" if val < 40 else "#f8d7da"
                    return [f"background-color: {color}"] * len(row)

                fmt = {}
                if "GI" in food_recs.columns: fmt["GI"] = "{:.0f}"
                if "GL" in food_recs.columns: fmt["GL"] = "{:.1f}"
                if "Carbs" in food_recs.columns: fmt["Carbs"] = "{:.1f}g"
                if "Protein" in food_recs.columns: fmt["Protein"] = "{:.1f}g"
                if "Fiber" in food_recs.columns: fmt["Fiber"] = "{:.1f}g"
                if "Predicted_Spike" in food_recs.columns: fmt["Predicted_Spike"] = "+{:.1f} mg/dL"
                if "Predicted_Peak" in food_recs.columns: fmt["Predicted_Peak"] = "{:.0f} mg/dL"
                if "Score" in food_recs.columns: fmt["Score"] = "{:.3f}"

                styled = food_recs[show_cols].style.apply(highlight_spike, axis=1).format(fmt)
                st.dataframe(styled, use_container_width=True)
            else:
                st.info("No food recommendations generated.")

            # ── MEAL PLAN ────────────────────────────────────────
            st.markdown("---")
            st.subheader("🍽️ Personalised Meal Plan")
            st.caption("Top 3 safe choices per meal from your food database.")

            if meal_plan:
                tabs = st.tabs(list(meal_plan.keys()))
                for tab, meal_name in zip(tabs, meal_plan.keys()):
                    with tab:
                        items = meal_plan[meal_name]
                        if items:
                            mdf = pd.DataFrame(items)
                            mfmt = {}
                            if "GI" in mdf.columns: mfmt["GI"] = "{:.0f}"
                            if "GL" in mdf.columns: mfmt["GL"] = "{:.1f}"
                            if "Predicted_Spike" in mdf.columns: mfmt["Predicted_Spike"] = "+{:.1f} mg/dL"
                            if "Score" in mdf.columns: mfmt["Score"] = "{:.3f}"
                            st.dataframe(mdf.style.format(mfmt), use_container_width=True)
                        else:
                            st.info(f"No items found for {meal_name}.")
            else:
                st.info("No meal plan generated.")

            # ── ACTIVITY ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("🏃 Activity Recommendation")

            urgency = activity.get("urgency_score", 0)
            alert = activity.get("clinical_alert", "")
            act_name = activity.get("activity", "")
            duration = activity.get("duration", "")
            timing = activity.get("timing", "")
            intensity = activity.get("intensity", "")
            calories = activity.get("calorie_burn", "")
            evidence = activity.get("evidence", "")

            if urgency >= 0.8:
                st.error(alert)
            elif urgency >= 0.5:
                st.warning(alert)
            else:
                st.success(alert)

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Activity", act_name)
            a2.metric("Duration", duration)
            a3.metric("Intensity", intensity)
            a4.metric("Calorie Burn", calories)

            st.info(f"⏰ **Timing:** {timing}")
            st.caption(f"📚 Evidence: {evidence}")

            # ── CLINICAL SUMMARY ─────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Clinical Summary")

            top_food, top_gi, top_spike = "N/A", "N/A", 0.0
            if isinstance(food_recs, pd.DataFrame) and not food_recs.empty:
                top_food = food_recs.iloc[0].get("Food_Name", "N/A")
                top_gi = food_recs.iloc[0].get("GI", "N/A")
                top_spike = food_recs.iloc[0].get("Predicted_Spike", 0.0)

            urgency_label = "IMMEDIATE" if risk_level == "HIGH" else "SOON" if risk_level == "MEDIUM" else "ROUTINE"

            st.code(f"""
CLINICAL SUMMARY — {result['timestamp']}
==========================================
Risk Level       : {risk_level} ({urgency_label})
Current Glucose  : {current:.0f} mg/dL
Predicted Peak   : {peak:.0f} mg/dL
Glucose Spike    : {spike:.1f} mg/dL
30-min Trend     : {trend:+.1f} mg/dL
Action Required  : {risk['requires_action']}

Top Food         : {top_food} (GI: {top_gi}, Est. Spike: +{top_spike:.1f} mg/dL)
Activity         : {act_name} — {duration}
Alert            : {alert}
""", language="text")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)
