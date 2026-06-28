import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

from engine import Config, ClinicalOrchestrator, safe_risk

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GlucoseGuard CDSS",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        .stApp { font-family: 'Inter', sans-serif; }
        .risk-high { 
            background-color: #dc3545; 
            color: white; 
            padding: 15px; 
            border-radius: 10px;
            font-weight: bold;
        }
        .risk-medium { 
            background-color: #ffc107; 
            color: #000; 
            padding: 15px; 
            border-radius: 10px;
            font-weight: bold;
        }
        .risk-low { 
            background-color: #28a745; 
            color: white; 
            padding: 15px; 
            border-radius: 10px;
            font-weight: bold;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { 
            height: 50px; 
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .segment-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4A90D9;
            margin-bottom: 10px;
        }
        .segment-label {
            font-weight: bold;
            color: #4A90D9;
        }
        .data-missing {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 8px;
            color: #856404;
        }
        .cgm-input-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .cgm-input-item {
            text-align: center;
        }
        .cgm-input-item label {
            font-size: 12px;
            color: #6c757d;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SAFETY HELPERS ==========
def safe_last_value(data, default=120.0):
    """Safely get last value from list, return default if empty"""
    if data and len(data) > 0:
        return data[-1]
    return default

def safe_ensure_data(data, fallback_generator=None):
    """Ensure data is non-empty, using fallback if needed"""
    if not data or len(data) == 0:
        st.warning("⚠️ CGM data missing — using fallback simulation")
        if fallback_generator:
            return fallback_generator()
        return generate_sample_cgm()
    return data

def generate_sample_cgm(n=10):
    """Generate realistic CGM sample data with exactly n readings"""
    base = 120 + np.random.randn(n) * 10
    for i in range(1, len(base)):
        base[i] = base[i-1] + np.random.randn() * 5
    # Add a meal spike in the middle
    if len(base) > 5:
        mid = len(base) // 2
        base[mid:mid+3] += 30 + np.random.randn(3) * 10
    return np.clip(base, 70, 250).tolist()

def generate_sample_cgm_fixed():
    """Generate a fixed (non-random) sample for consistency"""
    return [120, 122, 118, 121, 125, 128, 130, 135, 132, 129]

def validate_cgm_data(data, required_length=10):
    """Validate CGM data has exactly required_length readings"""
    if not data or len(data) == 0:
        return False, "No data provided"
    if len(data) != required_length:
        return False, f"Expected exactly {required_length} readings, got {len(data)}"
    # Check all values are within valid range
    invalid = [x for x in data if x < 40 or x > 400]
    if invalid:
        return False, f"Values must be between 40-400 mg/dL. Invalid: {invalid[:3]}..."
    return True, "Valid"

@st.cache_resource
def load_orchestrator():
    config = Config()
    return ClinicalOrchestrator(config)

def plot_predictions(cgm_data, preds, current, segment_labels=None):
    """Plot CGM data with predictions and optional segment labels"""
    fig = go.Figure()
    
    # Historical data
    times = [datetime.now() - timedelta(minutes=5*(len(cgm_data)-i)) 
             for i in range(len(cgm_data))]
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=times,
        y=cgm_data,
        mode='lines+markers',
        name='Historical CGM',
        line=dict(color='#4A90D9', width=2.5),
        marker=dict(size=8, color='#4A90D9')
    ))
    
    # Current point
    fig.add_trace(go.Scatter(
        x=[times[-1]],
        y=[current],
        mode='markers',
        name='Current',
        marker=dict(color='#FF6B6B', size=14, symbol='star')
    ))
    
    # Predictions
    future_times = [
        times[-1] + timedelta(minutes=30),
        times[-1] + timedelta(minutes=60),
        times[-1] + timedelta(minutes=120)
    ]
    
    pred_means = [preds['30min']['mean'], preds['60min']['mean'], preds['120min']['mean']]
    pred_lower = [preds['30min']['lower'], preds['60min']['lower'], preds['120min']['lower']]
    pred_upper = [preds['30min']['upper'], preds['60min']['upper'], preds['120min']['upper']]
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=[times[-1]] + future_times,
        y=[current] + pred_means,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        marker=dict(size=10, color='#FF6B6B')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=[times[-1]] + future_times + future_times[::-1] + [times[-1]],
        y=[current] + pred_upper + pred_lower[::-1] + [current],
        fill='toself',
        fillcolor='rgba(255,107,107,0.15)',
        line=dict(color='rgba(255,107,107,0)'),
        name='95% CI'
    ))
    
    # Add target zones
    fig.add_hrect(y0=70, y1=140, fillcolor='rgba(40,167,69,0.1)', line_width=0, name='Normal Range')
    fig.add_hrect(y0=0, y1=70, fillcolor='rgba(255,0,0,0.08)', line_width=0, name='Hypoglycemia')
    fig.add_hrect(y0=180, y1=300, fillcolor='rgba(255,165,0,0.08)', line_width=0, name='Hyperglycemia')
    
    fig.update_layout(
        title='Glucose Trend with 30/60/120 min Predictions',
        xaxis_title='Time',
        yaxis_title='Glucose (mg/dL)',
        yaxis_range=[40, 300],
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450
    )
    
    return fig

def render_cgm_slider_input():
    """Render 10 sliders for CGM input (best UX)"""
    st.markdown("### 📊 Enter 10 CGM Readings")
    st.caption("Adjust each reading using the sliders below")
    
    cols = st.columns(5)
    cgm_data = []
    
    for i in range(10):
        col_idx = i % 5
        with cols[col_idx]:
            val = st.slider(
                f"#{i+1}",
                min_value=40.0,
                max_value=250.0,
                value=120.0 + np.random.randn() * 10,
                step=1.0,
                key=f"cgm_slider_{i}",
                format="%.0f"
            )
            cgm_data.append(val)
    
    return cgm_data

def render_cgm_text_input():
    """Render text input with strict 10-reading validation"""
    st.markdown("### 📝 Enter 10 CGM Readings")
    st.caption("Enter exactly 10 comma-separated values (40-400 mg/dL)")
    
    default_values = generate_sample_cgm_fixed()
    default_str = ", ".join([str(int(v)) for v in default_values])
    
    cgm_input = st.text_area(
        "CGM readings (comma-separated)",
        value=default_str,
        height=60,
        help="Example: 120, 122, 118, 121, 125, 128, 130, 135, 132, 129"
    )
    
    # Parse and validate
    try:
        cgm_data = [float(x.strip()) for x in cgm_input.split(",") if x.strip()]
        is_valid, msg = validate_cgm_data(cgm_data, 10)
        
        if is_valid:
            st.success("✅ Valid: Exactly 10 readings")
            return cgm_data
        else:
            st.error(f"❌ {msg}")
            return None
    except Exception as e:
        st.error(f"❌ Invalid input: {str(e)}")
        return None

def render_cgm_quick_input():
    """Render quick preset buttons for common patterns"""
    st.markdown("### ⚡ Quick Presets")
    st.caption("Click a button to load a predefined CGM pattern")
    
    col1, col2, col3, col4 = st.columns(4)
    
    presets = {
        "Normal": [120, 122, 118, 121, 125, 128, 130, 135, 132, 129],
        "Rising": [120, 125, 130, 135, 140, 145, 150, 155, 160, 165],
        "Falling": [180, 175, 170, 165, 160, 155, 150, 145, 140, 135],
        "Spike": [120, 122, 118, 130, 165, 180, 175, 160, 145, 130]
    }
    
    with col1:
        if st.button("📈 Normal", use_container_width=True):
            st.session_state.cgm_data = presets["Normal"]
            st.session_state.cgm_input_source = "preset"
            st.rerun()
    
    with col2:
        if st.button("📈 Rising", use_container_width=True):
            st.session_state.cgm_data = presets["Rising"]
            st.session_state.cgm_input_source = "preset"
            st.rerun()
    
    with col3:
        if st.button("📉 Falling", use_container_width=True):
            st.session_state.cgm_data = presets["Falling"]
            st.session_state.cgm_input_source = "preset"
            st.rerun()
    
    with col4:
        if st.button("🎯 Spike", use_container_width=True):
            st.session_state.cgm_data = presets["Spike"]
            st.session_state.cgm_input_source = "preset"
            st.rerun()
    
    # Show current preset if loaded
    if 'cgm_input_source' in st.session_state and st.session_state.cgm_input_source == "preset":
        if 'cgm_data' in st.session_state and st.session_state.cgm_data:
            st.info(f"Loaded: {st.session_state.cgm_data}")
    
    return st.session_state.get('cgm_data', None)

def main():
    st.title("🩺 GlucoseGuard CDSS")
    st.caption("Clinical Decision Support System for Diabetes Management")
    
    # Initialize session state with safe defaults
    if 'analyze' not in st.session_state:
        st.session_state.analyze = False
    if 'cgm_data' not in st.session_state:
        st.session_state.cgm_data = generate_sample_cgm_fixed()
    if 'cgm_input_source' not in st.session_state:
        st.session_state.cgm_input_source = "default"
    if 'carbs' not in st.session_state:
        st.session_state.carbs = 0
    if 'protein' not in st.session_state:
        st.session_state.protein = 0
    if 'fat' not in st.session_state:
        st.session_state.fat = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        input_mode = st.radio(
            "Input Mode",
            ["🎛️ Sliders", "📝 Text Entry", "⚡ Quick Presets"],
            help="Sliders: Best UX | Text: Copy-paste | Presets: Quick testing"
        )
        
        st.divider()
        
        # Input based on mode
        if input_mode == "🎛️ Sliders":
            cgm_data = render_cgm_slider_input()
            st.session_state.cgm_data = cgm_data
            st.session_state.cgm_input_source = "sliders"
        
        elif input_mode == "📝 Text Entry":
            cgm_data = render_cgm_text_input()
            if cgm_data:
                st.session_state.cgm_data = cgm_data
                st.session_state.cgm_input_source = "text"
        
        else:  # Quick Presets
            cgm_data = render_cgm_quick_input()
            if cgm_data:
                st.session_state.cgm_data = cgm_data
        
        # Show current data
        if st.session_state.cgm_data:
            st.divider()
            st.markdown("### 📊 Current Data")
            st.write(f"Readings: {len(st.session_state.cgm_data)}")
            st.line_chart(st.session_state.cgm_data, height=100)
            current_val = safe_last_value(st.session_state.cgm_data)
            st.caption(f"Current: {current_val:.0f} mg/dL")
        
        st.divider()
        st.markdown("### 🍽️ Meal Context")
        carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=st.session_state.carbs, step=5)
        protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=st.session_state.protein, step=5)
        fat = st.number_input("Fat (g)", min_value=0, max_value=100, value=st.session_state.fat, step=5)
        
        st.session_state.carbs = carbs
        st.session_state.protein = protein
        st.session_state.fat = fat
        
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            # Validate before analysis
            cgm_data = st.session_state.cgm_data
            is_valid, msg = validate_cgm_data(cgm_data, 10)
            
            if is_valid:
                st.session_state.analyze = True
                st.rerun()
            else:
                st.error(f"❌ {msg}")
    
    # Main content - show input instructions if not analyzing
    if not st.session_state.get('analyze', False):
        st.info("👈 Select input mode in the sidebar and enter 10 CGM readings, then click 'Analyze'")
        
        # Show current data preview
        if st.session_state.cgm_data:
            with st.expander("📊 Current CGM Data Preview", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.cgm_data,
                    mode='lines+markers',
                    name='CGM',
                    marker=dict(size=10)
                ))
                fig.update_layout(height=250, title="CGM Readings")
                st.plotly_chart(fig, use_container_width=True)
        return
    
    # ========== ANALYZE ==========
    try:
        # SAFE: Ensure we have valid data before analysis
        cgm_data = st.session_state.cgm_data
        is_valid, msg = validate_cgm_data(cgm_data, 10)
        
        if not is_valid:
            st.error(f"❌ {msg}")
            st.session_state.analyze = False
            st.rerun()
            return
        
        orchestrator = load_orchestrator()
        
        with st.spinner("Analyzing glucose data..."):
            result = orchestrator.run(
                cgm_data,
                carbs=st.session_state.get('carbs', 0),
                protein=st.session_state.get('protein', 0),
                fat=st.session_state.get('fat', 0),
                top_k=10
            )
        
        # SAFE: Extract risk with defaults
        risk = safe_risk(result.get('risk', {}))
        preds = result.get('predictions', {
            '30min': {'mean': 0, 'lower': 0, 'upper': 0},
            '60min': {'mean': 0, 'lower': 0, 'upper': 0},
            '120min': {'mean': 0, 'lower': 0, 'upper': 0}
        })
        current = risk.get('current', safe_last_value(cgm_data))
        
        # Risk banner
        risk_colors = {
            'HIGH': 'risk-high',
            'MEDIUM': 'risk-medium', 
            'LOW': 'risk-low'
        }
        risk_level = risk.get('risk_level', 'LOW')
        dominant_risk = risk.get('dominant_risk', 'NONE')
        clinical_summary = risk.get('clinical_summary', 'No summary available')
        
        st.markdown(f"""
            <div class="{risk_colors.get(risk_level, 'risk-low')}">
                🚨 RISK: {risk_level} — {dominant_risk}
                <br><small>{clinical_summary}</small>
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics row with safe values
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Current", f"{current:.0f} mg/dL")
        with col2:
            st.metric("30-min", f"{preds['30min']['mean']:.0f}", 
                     delta=f"{preds['30min']['mean'] - current:.0f}")
        with col3:
            st.metric("60-min", f"{preds['60min']['mean']:.0f}",
                     delta=f"{preds['60min']['mean'] - current:.0f}")
        with col4:
            st.metric("120-min", f"{preds['120min']['mean']:.0f}",
                     delta=f"{preds['120min']['mean'] - current:.0f}")
        with col5:
            velocity = risk.get('glucose_velocity', 0.0)
            st.metric("Velocity", f"{velocity:.2f} mg/dL/min")
        
        # Plot with segment labels if available
        st.plotly_chart(plot_predictions(cgm_data, preds, current), use_container_width=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Glucose Predictions",
            "🍽️ Food Recommendations",
            "🏃 Activity Plan",
            "📋 Clinical Summary"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 Prediction Details")
                pred_df = pd.DataFrame({
                    'Horizon': ['30 min', '60 min', '120 min'],
                    'Mean': [preds['30min']['mean'], preds['60min']['mean'], preds['120min']['mean']],
                    'Lower (95% CI)': [preds['30min']['lower'], preds['60min']['lower'], preds['120min']['lower']],
                    'Upper (95% CI)': [preds['30min']['upper'], preds['60min']['upper'], preds['120min']['upper']]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### 🎯 Risk Metrics")
                st.metric("Peak", f"{risk.get('peak', 0):.0f} mg/dL")
                st.metric("Trough", f"{risk.get('trough', 0):.0f} mg/dL")
                st.metric("Risk Score", f"{risk.get('risk_score', 0):.2f}")
                st.metric("Requires Action", "✅ Yes" if risk.get('requires_action', False) else "❌ No")
                
                # Show input source
                st.markdown("### 📊 Input Info")
                st.write(f"- **Mode:** {st.session_state.cgm_input_source}")
                st.write(f"- **Readings:** {len(cgm_data)}")
                st.write(f"- **Range:** {min(cgm_data):.0f} - {max(cgm_data):.0f} mg/dL")
        
        with tab2:
            food_recs = result.get('food_recommendations')
            if food_recs is not None and not food_recs.empty:
                st.markdown("### 🍽️ Top Food Recommendations")
                
                meal_plan = result.get('meal_plan', {})
                for meal, items in meal_plan.items():
                    if items:
                        with st.expander(f"🥗 {meal}", expanded=True):
                            df = pd.DataFrame(items)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                
                with st.expander("📋 All Recommendations", expanded=False):
                    st.dataframe(food_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("No food recommendations available")
        
        with tab3:
            activity = result.get('activity', {
                'activity': 'No recommendation',
                'duration': 'N/A',
                'timing': 'N/A',
                'intensity': 'N/A',
                'calorie_burn': 'N/A',
                'clinical_alert': 'No alert',
                'evidence': 'No evidence available',
                'urgency_score': 0
            })
            st.markdown("### 🏃 Activity Recommendation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Activity:** {activity.get('activity', 'N/A')}")
                st.info(f"**Duration:** {activity.get('duration', 'N/A')}")
                st.info(f"**Timing:** {activity.get('timing', 'N/A')}")
            with col2:
                st.info(f"**Intensity:** {activity.get('intensity', 'N/A')}")
                st.info(f"**Calorie Burn:** {activity.get('calorie_burn', 'N/A')}")
                st.warning(f"**Clinical Alert:** {activity.get('clinical_alert', 'No alert')}")
                st.caption(f"**Evidence:** {activity.get('evidence', 'No evidence')}")
        
        with tab4:
            st.markdown("### 📋 Clinical Summary")
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("#### Patient State")
                st.write(f"- **Current Glucose:** {current:.0f} mg/dL")
                st.write(f"- **Risk Level:** {risk_level}")
                st.write(f"- **Dominant Risk:** {dominant_risk}")
                st.write(f"- **Trend Direction:** {risk.get('trend_direction', 'STABLE')}")
                st.write(f"- **Glucose Velocity:** {risk.get('glucose_velocity', 0):.2f} mg/dL/min")
            
            with cols[1]:
                st.markdown("#### Predicted Events")
                st.write(f"- **Predicted Peak:** {risk.get('peak', 0):.0f} mg/dL")
                st.write(f"- **Predicted Trough:** {risk.get('trough', 0):.0f} mg/dL")
                st.write(f"- **Spike Magnitude:** {risk.get('upward_spike', 0):.0f} mg/dL")
                st.write(f"- **Drop Magnitude:** {risk.get('downward_drop', 0):.0f} mg/dL")
            
            st.markdown("#### Clinical Interpretation")
            st.info(clinical_summary)
            
            # Meal context
            st.markdown("#### Meal Context")
            st.write(f"- **Carbs:** {st.session_state.carbs}g")
            st.write(f"- **Protein:** {st.session_state.protein}g")
            st.write(f"- **Fat:** {st.session_state.fat}g")
        
        # Reset button
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analyze = False
            st.session_state.cgm_data = generate_sample_cgm_fixed()
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)
        st.session_state.analyze = False

if __name__ == "__main__":
    main()
