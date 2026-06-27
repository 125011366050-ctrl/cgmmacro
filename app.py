import streamlit as st
import numpy as np
import pandas as pd
from reco import ClinicalOrchestrator, Config

st.set_page_config(page_title="CDSS AI System", page_icon="🩸")

st.title("🩸 Diabetes CDSS Recommendation System")

@st.cache_resource
def load_system():
    config = Config()
    system = ClinicalOrchestrator(config)
    return system

system = load_system()

st.header("Patient Input")

glucose = st.number_input("Current Glucose", 40.0, 400.0, 120.0)
heart_rate = st.number_input("Heart Rate", 40.0, 180.0, 80.0)
calories = st.number_input("Calories", 0.0, 2000.0, 250.0)
carbs = st.number_input("Carbs", 0.0, 300.0, 40.0)
protein = st.number_input("Protein", 0.0, 200.0, 20.0)
fat = st.number_input("Fat", 0.0, 200.0, 10.0)
fiber = st.number_input("Fiber", 0.0, 100.0, 5.0)

meal_flag = 1

# dummy window (IMPORTANT)
def create_window(glucose):
    window_size = 36
    features = 18
    base = np.zeros((window_size, features), dtype=np.float32)

    base[:, 0] = glucose  # GL
    base[:, 7] = heart_rate
    base[:, 10] = calories
    base[:, 11] = carbs
    base[:, 12] = protein
    base[:, 13] = fat
    base[:, 14] = fiber
    base[:, 9] = meal_flag

    return base

if st.button("Predict & Recommend"):

    x_window = create_window(glucose)

    result = system.run(
        current_glucose=glucose,
        x_window=x_window,
        top_k=5
    )

    st.subheader("📊 Risk Analysis")
    st.write(result["risk"])

    st.subheader("📈 Predictions")
    st.write(result["predictions"])

    st.subheader("🏃 Activity Recommendation")
    st.write(result["activity"])

    st.subheader("🍛 Food Recommendations")

    df = pd.DataFrame(result["food_recommendations"])
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("No food recommendations available")

    st.subheader("🍽️ Meal Plan")
    st.json(result["meal_plan"])

    st.success("Prediction completed")
