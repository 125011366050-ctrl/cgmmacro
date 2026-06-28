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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_orchestrator():
    config = Config()
    return ClinicalOrchestrator(config)

def generate_sample_cgm():
    base = 120 + np.random.randn(50) * 10
    for i in range(1, len(base)):
        base[i] = base[i-1] + np.random.randn() * 5
    # Add a meal spike
    if len(base) > 30:
        base[25:35] += 30 + np.random.randn(10) * 10
    return np.clip(base, 70, 250).tolist()

def generate_segment_cgm(segment_type="Fasting", length=20):
    """Generate realistic CGM segment based on type"""
    if segment_type == "Fasting":
        base = 100 + np.random.randn(length) * 8
        for i in range(1, len(base)):
            base[i] = base[i-1] + np.random.randn() * 2
    elif segment_type == "Breakfast":
        base = 100 + np.random.randn(length) * 8
        for i in range(1, len(base)):
            base[i] = base[i-1] + np.random.randn() * 2
        if length > 5:
            base[5:15] += 40 + np.random.randn(10) * 15
    elif segment_type == "Lunch":
        base = 120 + np.random.randn(length) * 10
        for i in range(1, len(base)):
            base[i] = base[i-1] + np.random.randn() * 2
        if length > 5:
            base[5:15] += 50 + np.random.randn(10) * 15
    elif segment_type == "Dinner":
        base = 130 + np.random.randn(length) * 10
        for i in range(1, len(base)):
            base[i] = base[i-1] + np.random.randn() * 2
        if length > 5:
            base[5:15] += 45 + np.random.randn(10) * 15
    elif segment_type == "Snack":
        base = 110 + np.random.randn(length) * 8
        for i in range(1, len(base)):
            base[i] = base[i-1] + np.random.randn() * 2
        if length > 5:
            base[5:10] += 25 + np.random.randn(5) * 10
    else:  # Sleep
        base = 100 + np.random.randn(length) * 5
        for i in range(1, len(base)):
            base[i] = base[i-1] + np.random.randn() * 1
    
    return np.clip(base, 70, 250).tolist()

def plot_predictions(cgm_data, preds, current, segment_labels=None):
    fig = go.Figure()
    
    # Historical data
    times = [datetime.now() - timedelta(minutes=5*(len(cgm_data)-i)) 
             for i in range(len(cgm_data))]
    
    # Add segment backgrounds if labels provided
    if segment_labels:
        segment_colors = {
            "Fasting": "rgba(100, 149, 237, 0.1)",
            "Breakfast": "rgba(255, 165, 0, 0.1)",
            "Lunch": "rgba(255, 99, 71, 0.1)",
            "Dinner": "rgba(128, 0, 128, 0.1)",
            "Snack": "rgba(60, 179, 113, 0.1)",
            "Sleep": "rgba(70, 130, 180, 0.1)"
        }
        
        # Group points by segment
        segment_changes = []
        for i in range(1, len(segment_labels)):
            if segment_labels[i] != segment_labels[i-1]:
                segment_changes.append(i)
        
        if segment_changes:
            start_idx = 0
            for end_idx in segment_changes + [len(cgm_data)]:
                label = segment_labels[start_idx] if start_idx < len(segment_labels) else "Unknown"
                fig.add_vrect(
                    x0=times[start_idx],
                    x1=times[end_idx-1] if end_idx < len(times) else times[-1],
                    fillcolor=segment_colors.get(label, "rgba(200, 200, 200, 0.1)"),
                    line_width=0,
                    label=dict(text=label, font_size=10, font_color="gray")
                )
                start_idx = end_idx
    
    fig.add_trace(go.Scatter(
        x=times,
        y=cgm_data,
        mode='lines',
        name='Historical CGM',
        line=dict(color='#4A90D9', width=2.5)
    ))
    
    # Current point
    fig.add_trace(go.Scatter(
        x=[times[-1]],
        y=[current],
        mode='markers',
        name='Current',
        marker=dict(color='#FF6B6B', size=12, symbol='star')
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
        marker=dict(size=8, color='#FF6B6B')
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

def render_segment_inputs():
    """Render multiple CGM segment inputs"""
    st.markdown("### 🧪 Multiple CGM Inputs")
    st.caption("Add multiple CGM segments to simulate real-world glucose patterns (fasting → post-meal → recovery)")
    
    n_segments = st.number_input(
        "Number of CGM segments",
        min_value=1,
        max_value=5,
        value=2,
        help="Split your CGM data into meaningful segments (e.g., fasting, breakfast, lunch, dinner)"
    )
    
    cgm_segments = []
    segment_labels = []
    all_valid = True
    
    for i in range(n_segments):
        st.markdown(f"""
        <div class="segment-card">
            <span class="segment-label">Segment {i+1}</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            segment_type = st.selectbox(
                f"Type",
                ["Fasting", "Breakfast", "Lunch", "Dinner", "Snack", "Sleep"],
                key=f"seg_type_{i}",
                index=0
            )
            
            # Meal context for this segment
            if segment_type != "Fasting" and segment_type != "Sleep":
                segment_carbs = st.number_input(
                    f"Carbs (g)",
                    min_value=0, max_value=150, value=30, step=5,
                    key=f"seg_carbs_{i}"
                )
            else:
                segment_carbs = 0
                st.caption("Fasting/Sleep: No meal context")
        
        with col2:
            # Pre-populate based on segment type
            default_values = generate_segment_cgm(segment_type, length=15)
            default_str = ", ".join([f"{int(v)}" for v in default_values])
            
            input_text = st.text_area(
                f"CGM values (comma-separated)",
                value=default_str,
                key=f"seg_input_{i}",
                height=80,
                help="Enter at least 5 readings for this segment"
            )
            
            try:
                values = [float(x.strip()) for x in input_text.split(",") if x.strip()]
                if len(values) < 5:
                    st.warning(f"⚠️ Segment {i+1} too short (min 5 required) — using generated values")
                    values = generate_segment_cgm(segment_type, length=15)
                    all_valid = False
                else:
                    st.success(f"✅ {len(values)} readings loaded")
            except Exception as e:
                st.error(f"❌ Invalid input — using generated sample")
                values = generate_segment_cgm(segment_type, length=15)
                all_valid = False
        
        cgm_segments.append(values)
        segment_labels.append(segment_type)
        
        # Store meal context for each segment
        if segment_type not in ["Fasting", "Sleep"]:
            st.session_state[f"seg_{i}_carbs"] = segment_carbs
        else:
            st.session_state[f"seg_{i}_carbs"] = 0
        
        st.divider()
    
    return cgm_segments, segment_labels, all_valid

def main():
    st.title("🩺 GlucoseGuard CDSS")
    st.caption("Clinical Decision Support System for Diabetes Management")
    
    # Initialize session state
    if 'analyze' not in st.session_state:
        st.session_state.analyze = False
    if 'cgm_data' not in st.session_state:
        st.session_state.cgm_data = []
    if 'segment_labels' not in st.session_state:
        st.session_state.segment_labels = []
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Multiple Segments"
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Cleaner input mode selection
        input_mode = st.radio(
            "Input Mode",
            ["🧩 Multiple Segments", "📊 Single Segment", "📝 Manual Entry"],
            help="Multiple Segments: Split CGM into fasting/meal segments | Single: Continuous stream | Manual: Custom entry"
        )
        st.session_state.input_mode = input_mode
        
        if input_mode == "🧩 Multiple Segments":
            # Multi-segment input is rendered in main area
            st.info("Configure segments in the main panel")
            
            # We'll pass the segments to main
            if st.button("🚀 Analyze All Segments", type="primary", use_container_width=True):
                st.session_state.analyze = True
                st.rerun()
        
        elif input_mode == "📊 Single Segment":
            # Simple single CGM stream
            if st.button("🔄 Generate Sample"):
                st.session_state.cgm_data = generate_sample_cgm()
                st.session_state.segment_labels = ["Fasting"] * len(st.session_state.cgm_data)
                st.rerun()
            
            cgm_data = st.session_state.get('cgm_data', generate_sample_cgm())
            st.line_chart(cgm_data, height=150)
            st.caption(f"Current: {cgm_data[-1]:.0f} mg/dL")
            
            # Meal context
            st.divider()
            st.markdown("### 🍽️ Meal Context")
            carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=0, step=5)
            protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=0, step=5)
            fat = st.number_input("Fat (g)", min_value=0, max_value=100, value=0, step=5)
            
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                st.session_state.analyze = True
                st.session_state.cgm_data = cgm_data
                st.session_state.carbs = carbs
                st.session_state.protein = protein
                st.session_state.fat = fat
                st.rerun()
        
        else:  # Manual Entry
            cgm_input = st.text_area(
                "Enter CGM readings (comma-separated)",
                "120, 122, 118, 121, 125, 128, 130, 135, 132, 129, 126, 124, 127, 131, 134, 138, 142, 145, 148, 150"
            )
            try:
                cgm_data = [float(x.strip()) for x in cgm_input.split(',') if x.strip()]
                if len(cgm_data) < 5:
                    st.warning("Please enter at least 5 readings")
                    cgm_data = generate_sample_cgm()
            except:
                st.error("Invalid input — using sample data")
                cgm_data = generate_sample_cgm()
            
            st.caption(f"Current: {cgm_data[-1]:.0f} mg/dL")
            st.session_state.cgm_data = cgm_data
            
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                st.session_state.analyze = True
                st.rerun()
    
    # Main content
    if st.session_state.input_mode == "🧩 Multiple Segments" and not st.session_state.get('analyze', False):
        # Render segment inputs
        cgm_segments, segment_labels, valid = render_segment_inputs()
        
        if valid and len(cgm_segments) > 0:
            st.success(f"✅ {len(cgm_segments)} segments loaded successfully")
            
            # Show preview
            with st.expander("📊 Preview CGM Data"):
                # Merge for preview
                merged = np.concatenate(cgm_segments).tolist()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=merged,
                    mode='lines',
                    name='Merged CGM'
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show segment boundaries
                boundaries = [0]
                for seg in cgm_segments:
                    boundaries.append(boundaries[-1] + len(seg))
                
                st.write("**Segment lengths:**")
                for i, (label, seg) in enumerate(zip(segment_labels, cgm_segments)):
                    st.write(f"- {label}: {len(seg)} readings")
            
            if st.button("🚀 Analyze Multi-Segment Data", type="primary", use_container_width=True):
                st.session_state.analyze = True
                st.session_state.cgm_data = np.concatenate(cgm_segments).tolist()
                st.session_state.segment_labels = []
                for i, label in enumerate(segment_labels):
                    st.session_state.segment_labels.extend([label] * len(cgm_segments[i]))
                st.session_state.carbs = 0
                st.session_state.protein = 0
                st.session_state.fat = 0
                st.rerun()
        else:
            st.warning("⚠️ Please fix all segment inputs before analyzing")
        return
    
    # Handle analysis
    if not st.session_state.get('analyze', False):
        st.info("👈 Configure input in the sidebar and click 'Analyze'")
        return
    
    try:
        orchestrator = load_orchestrator()
        cgm_data = st.session_state.cgm_data
        
        if len(cgm_data) < 5:
            st.error("❌ Not enough CGM data (need at least 5 readings)")
            st.session_state.analyze = False
            return
        
        with st.spinner("Analyzing glucose data..."):
            result = orchestrator.run(
                cgm_data,
                carbs=st.session_state.get('carbs', 0),
                protein=st.session_state.get('protein', 0),
                fat=st.session_state.get('fat', 0),
                top_k=10
            )
        
        risk = safe_risk(result['risk'])
        preds = result['predictions']
        current = risk['current']
        segment_labels = st.session_state.get('segment_labels', [])
        
        # Risk banner
        risk_colors = {
            'HIGH': 'risk-high',
            'MEDIUM': 'risk-medium', 
            'LOW': 'risk-low'
        }
        st.markdown(f"""
            <div class="{risk_colors.get(risk['risk_level'], 'risk-low')}">
                🚨 RISK: {risk['risk_level']} — {risk['dominant_risk']}
                <br><small>{risk['clinical_summary']}</small>
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics row
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
            st.metric("Velocity", f"{risk['glucose_velocity']:.2f} mg/dL/min")
        
        # Plot with segment labels
        st.plotly_chart(plot_predictions(cgm_data, preds, current, segment_labels), use_container_width=True)
        
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
                st.metric("Peak", f"{risk['peak']:.0f} mg/dL")
                st.metric("Trough", f"{risk['trough']:.0f} mg/dL")
                st.metric("Risk Score", f"{risk['risk_score']:.2f}")
                st.metric("Requires Action", "✅ Yes" if risk['requires_action'] else "❌ No")
                
                # Show segment information
                if segment_labels:
                    st.markdown("### 📂 Segments")
                    unique_labels, counts = np.unique(segment_labels, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        st.write(f"- {label}: {count} readings")
        
        with tab2:
            if 'food_recommendations' in result and not result['food_recommendations'].empty:
                st.markdown("### 🍽️ Top Food Recommendations")
                
                # Filter by meal plan
                meal_plan = result.get('meal_plan', {})
                for meal, items in meal_plan.items():
                    if items:
                        with st.expander(f"🥗 {meal}", expanded=True):
                            df = pd.DataFrame(items)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Full recommendations
                with st.expander("📋 All Recommendations", expanded=False):
                    st.dataframe(result['food_recommendations'], use_container_width=True, hide_index=True)
            else:
                st.warning("No food recommendations available")
        
        with tab3:
            activity = result['activity']
            st.markdown("### 🏃 Activity Recommendation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Activity:** {activity['activity']}")
                st.info(f"**Duration:** {activity['duration']}")
                st.info(f"**Timing:** {activity['timing']}")
            with col2:
                st.info(f"**Intensity:** {activity['intensity']}")
                st.info(f"**Calorie Burn:** {activity['calorie_burn']}")
                st.warning(f"**Clinical Alert:** {activity['clinical_alert']}")
                st.caption(f"**Evidence:** {activity['evidence']}")
        
        with tab4:
            st.markdown("### 📋 Clinical Summary")
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("#### Patient State")
                st.write(f"- **Current Glucose:** {current:.0f} mg/dL")
                st.write(f"- **Risk Level:** {risk['risk_level']}")
                st.write(f"- **Dominant Risk:** {risk['dominant_risk']}")
                st.write(f"- **Trend Direction:** {risk['trend_direction']}")
                
                # Show segment breakdown
                if segment_labels:
                    st.markdown("#### Segment Analysis")
                    st.write(f"- **Segments:** {len(np.unique(segment_labels))}")
                    st.write(f"- **Total Readings:** {len(segment_labels)}")
            
            with cols[1]:
                st.markdown("#### Predicted Events")
                st.write(f"- **Predicted Peak:** {risk['peak']:.0f} mg/dL")
                st.write(f"- **Predicted Trough:** {risk['trough']:.0f} mg/dL")
                st.write(f"- **Spike Magnitude:** {risk['upward_spike']:.0f} mg/dL")
                st.write(f"- **Drop Magnitude:** {risk['downward_drop']:.0f} mg/dL")
            
            st.markdown("#### Clinical Interpretation")
            st.info(risk['clinical_summary'])
        
        # Reset button
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analyze = False
            st.session_state.cgm_data = []
            st.session_state.segment_labels = []
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)
        st.session_state.analyze = False

if __name__ == "__main__":
    main()
