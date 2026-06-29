import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import copy
import inspect

# Import from engine
from engine import Config, ClinicalOrchestrator

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GlucoseGuard CDSS",
    page_icon="🩺",
    layout="wide"
)

# ========== CONSTANTS ==========
CGM_MIN = 40.0
CGM_MAX = 250.0
REQUIRED_READINGS = 10
CGM_INTERVAL_MINUTES = 5
VELOCITY_THRESHOLD = 0.5
HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0

# ========== STYLES ==========
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        .stApp { font-family: 'Inter', sans-serif; }
        .risk-high { background-color: #dc3545; color: white; padding: 15px; border-radius: 10px; font-weight: bold; }
        .risk-medium { background-color: #ffc107; color: #000; padding: 15px; border-radius: 10px; font-weight: bold; }
        .risk-low { background-color: #28a745; color: white; padding: 15px; border-radius: 10px; font-weight: bold; }
        .risk-unknown { background-color: #6c757d; color: white; padding: 15px; border-radius: 10px; font-weight: bold; }
        .risk-override { background-color: #8B0000; color: white; padding: 15px; border-radius: 10px; border: 2px solid #ff0000; font-weight: bold; }
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ========== SCHEMA VALIDATION ==========
def validate_engine_output(result: Any) -> Tuple[bool, str]:
    """Validate engine output has required structure"""
    if not result:
        return False, "No result returned"
    if not isinstance(result, dict):
        return False, "Result is not a dictionary"
    
    required = ["risk", "predictions"]
    missing = [k for k in required if k not in result]
    if missing:
        return False, f"Missing required keys: {missing}"
    
    return True, "Valid"

# ========== CLINICAL REASONING ENGINE ==========
class ClinicalReasoningEngine:
    """Separate clinical reasoning engine with uncertainty awareness"""
    
    def __init__(self):
        self.reasoning_history = []
    
    def generate_reasoning(self, risk: Dict, preds: Dict, current: float, 
                          reliability: Dict, velocity: float, 
                          trust: Dict, clinical_rules: List[str] = None) -> Dict:
        """Generate comprehensive clinical reasoning"""
        
        risk_level = risk.get('risk_level', 'LOW')
        peak = risk.get('peak', current)
        trough = risk.get('trough', current)
        
        parts = []
        confidence_modifiers = []
        actionable = False
        rule_overrides = []
        
        # Uncertainty-aware reasoning
        if reliability['label'] == 'LOW':
            confidence_modifiers.append("⚠️ Prediction uncertainty is HIGH; interpret with caution")
            parts.append("Model confidence is low - clinical judgment recommended")
        elif reliability['label'] == 'MEDIUM':
            confidence_modifiers.append("⚡ Prediction uncertainty is MODERATE")
        else:
            confidence_modifiers.append("✅ Prediction uncertainty is LOW")
        
        # Trust score
        if trust['label'] == 'LOW':
            confidence_modifiers.append(f"⚠️ Model trust is LOW ({trust['score']:.0%})")
            parts.append("Model-data disagreement detected - verify trend")
        
        # Trend analysis
        if abs(velocity) > 2.0:
            direction = "rising rapidly" if velocity > 0 else "falling rapidly"
            parts.append(f"Glucose is {direction} at {abs(velocity):.2f} mg/dL/min")
            actionable = True
        elif abs(velocity) > 1.0:
            direction = "rising" if velocity > 0 else "falling"
            parts.append(f"Glucose is {direction} at {abs(velocity):.2f} mg/dL/min")
        else:
            parts.append("Glucose is stable")
        
        # Risk-specific reasoning
        if risk_level == "HIGH":
            if peak > HYPER_THRESHOLD:
                parts.append(f"Predicted peak of {peak:.0f} mg/dL indicates hyperglycemia risk")
            if trough < HYPO_THRESHOLD:
                parts.append(f"Predicted trough of {trough:.0f} mg/dL indicates hypoglycemia risk")
            parts.append("🚨 Immediate clinical attention recommended")
            actionable = True
        elif risk_level == "MEDIUM":
            if peak > 140:
                parts.append(f"Predicted peak of {peak:.0f} mg/dL above normal range")
            parts.append("⚡ Monitor glucose closely")
            actionable = True
        elif risk_level == "UNKNOWN":
            parts.append("⚠️ Model returned unknown risk level - clinical interpretation limited")
            actionable = True
        else:
            parts.append("Glucose within acceptable range")
        
        # Clinical rules
        if clinical_rules:
            rule_overrides = clinical_rules
            parts.extend(clinical_rules)
            actionable = True
        
        reasoning = " ".join(parts)
        full_reasoning = f"{reasoning}\n\n{chr(10).join(confidence_modifiers)}"
        
        self.reasoning_history.append({
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level,
            'reasoning': full_reasoning,
            'actionable': actionable,
            'rule_overrides': rule_overrides
        })
        
        return {
            'summary': reasoning,
            'full_reasoning': full_reasoning,
            'confidence_modifier': " ".join(confidence_modifiers),
            'actionable': actionable,
            'rule_overrides': rule_overrides,
            'history': self.reasoning_history
        }

# ========== PREDICTION AUDIT ==========
class PredictionAudit:
    """Single source of truth for audit trail"""
    
    def __init__(self):
        self._log = []
    
    def log_prediction(self, cgm_data: List[float], preds: Dict, risk: Dict, 
                       velocity: float, reliability: Dict, trust: Dict,
                       engine_signature: str, model_version: str = "v1.0",
                       clinical_rules: List[str] = None) -> None:
        """Log prediction for audit trail"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'cgm_input': cgm_data,
            'predictions': preds,
            'risk': risk,
            'velocity': velocity,
            'reliability': reliability,
            'trust': trust,
            'engine_signature': engine_signature,
            'model_version': model_version,
            'clinical_rules': clinical_rules or []
        }
        self._log.append(entry)
        st.session_state['prediction_audit'] = self._log.copy()
    
    def get_last(self) -> Optional[Dict]:
        return self._log[-1] if self._log else None
    
    def get_history(self) -> List[Dict]:
        return self._log.copy()
    
    def export_audit(self) -> str:
        return json.dumps(self._log, default=str, indent=2)
    
    def clear(self):
        self._log = []
        st.session_state['prediction_audit'] = []

# ========== RISK NORMALIZER ==========
class RiskNormalizer:
    @staticmethod
    def normalize(risk: Any) -> Dict:
        default = {
            "risk_level": "LOW",
            "dominant_risk": "NONE",
            "clinical_summary": "No summary available",
            "current": None,
            "glucose_velocity": 0.0,
            "trend_direction": "STABLE",
            "trend_strength": 0.0,
            "peak": 0.0,
            "trough": 0.0,
            "upward_spike": 0.0,
            "downward_drop": 0.0,
            "drop_severity": "NORMAL",
            "requires_action": False,
            "risk_score": 0.0,
            "predictions": [],
            "uncertainty": {"lower": [], "upper": [], "std": []}
        }
        if not isinstance(risk, dict):
            return default
        
        normalized = copy.deepcopy(default)
        for k in default.keys():
            if k in risk and risk[k] is not None:
                normalized[k] = risk[k]
        
        if "risk_level" in normalized:
            normalized["risk_level"] = RiskNormalizer._normalize_level(normalized["risk_level"])
        
        return normalized
    
    @staticmethod
    def _normalize_level(level: str) -> str:
        if not level:
            return "LOW"
        
        level = str(level).strip().upper()
        for sep in [" ", "_", "-", "–", "—", ".", ","]:
            level = level.replace(sep, "")
        
        mapping = {
            "HIGH": "HIGH", "HIGHRISK": "HIGH", "SEVERE": "HIGH",
            "CRITICAL": "HIGH", "DANGEROUS": "HIGH", "EMERGENCY": "HIGH",
            "MEDIUM": "MEDIUM", "MODERATE": "MEDIUM", "ELEVATED": "MEDIUM",
            "LOW": "LOW", "NORMAL": "LOW", "STABLE": "LOW", "GOOD": "LOW", "NONE": "LOW"
        }
        
        if level not in mapping:
            if 'risk_warnings' not in st.session_state:
                st.session_state.risk_warnings = []
            
            if not any(w['unknown_risk'] == level for w in st.session_state.risk_warnings):
                st.session_state.risk_warnings.append({
                    'timestamp': datetime.now().isoformat(),
                    'unknown_risk': level,
                    'mapped_to': 'UNKNOWN'
                })
            
            return "UNKNOWN"
        return mapping[level]

# ========== PREDICTION NORMALIZER ==========
class PredictionNormalizer:
    @staticmethod
    def normalize(preds: Any) -> Tuple[Optional[Dict], bool]:
        default = {
            '30min': {'mean': 0.0, 'lower': 0.0, 'upper': 0.0},
            '60min': {'mean': 0.0, 'lower': 0.0, 'upper': 0.0},
            '120min': {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
        }
        
        if not isinstance(preds, dict):
            return None, False
        
        has_valid_values = False
        normalized = copy.deepcopy(default)
        is_valid = True
        
        for k in default.keys():
            if k not in preds or not isinstance(preds[k], dict):
                is_valid = False
                continue
            
            for sub in ['mean', 'lower', 'upper']:
                if sub not in preds[k] or preds[k][sub] is None:
                    is_valid = False
                    normalized[k][sub] = default[k][sub]
                else:
                    try:
                        value = float(preds[k][sub])
                        if np.isnan(value) or np.isinf(value):
                            is_valid = False
                            normalized[k][sub] = default[k][sub]
                        else:
                            normalized[k][sub] = value
                            if abs(value) > 0.01:
                                has_valid_values = True
                    except (ValueError, TypeError):
                        is_valid = False
                        normalized[k][sub] = default[k][sub]
        
        if not has_valid_values:
            return None, False
        
        return normalized, is_valid

# ========== CLINICAL GUARD ==========
class ClinicalGuard:
    @staticmethod
    def normalize_activity(activity: Any) -> Dict:
        defaults = {
            'activity': 'No recommendation',
            'duration': 'N/A',
            'timing': 'N/A',
            'intensity': 'N/A',
            'calorie_burn': 'N/A',
            'clinical_alert': 'No alert',
            'evidence': 'No evidence available',
            'urgency_score': 0
        }
        if not isinstance(activity, dict):
            return defaults
        normalized = copy.deepcopy(defaults)
        for k in defaults.keys():
            if k in activity and activity[k] is not None:
                normalized[k] = activity[k]
        return normalized
    
    @staticmethod
    def normalize_food_recs(food_recs: Any) -> Optional[pd.DataFrame]:
        if food_recs is None:
            return None
        if isinstance(food_recs, pd.DataFrame):
            return food_recs if not food_recs.empty else None
        if isinstance(food_recs, list):
            return pd.DataFrame(food_recs) if food_recs else None
        return None
    
    @staticmethod
    def normalize_meal_plan(meal_plan: Any) -> Dict:
        return meal_plan if isinstance(meal_plan, dict) else {}

# ========== CLINICAL RULE ENGINE ==========
class ClinicalRuleEngine:
    @staticmethod
    def apply_rules(cgm_data: List[float], preds: Dict, risk: Dict) -> Tuple[Dict, List[str]]:
        if not cgm_data or len(cgm_data) == 0:
            return risk, []
        
        current = cgm_data[-1]
        rules_triggered = []
        override_risk = None
        
        if current < HYPO_THRESHOLD:
            override_risk = "HIGH"
            rules_triggered.append(f"🚨 CLINICAL RULE: Current glucose {current:.0f} mg/dL below {HYPO_THRESHOLD:.0f} mg/dL threshold - HIGH risk override")
        
        if current < 55:
            override_risk = "HIGH"
            rules_triggered.append(f"🚨 CRITICAL RULE: Current glucose {current:.0f} mg/dL - SEVERE hypoglycemia")
        
        if current > HYPER_THRESHOLD:
            if override_risk is None or override_risk == "LOW":
                override_risk = "HIGH"
            rules_triggered.append(f"⚠️ CLINICAL RULE: Current glucose {current:.0f} mg/dL above {HYPER_THRESHOLD:.0f} mg/dL threshold - HIGH risk override")
        
        if len(cgm_data) >= 3:
            fall = cgm_data[-1] - cgm_data[-3]
            if fall < -30:
                if override_risk is None or override_risk == "LOW":
                    override_risk = "MEDIUM"
                rules_triggered.append(f"⚠️ CLINICAL RULE: Rapid fall of {abs(fall):.0f} mg/dL detected - MEDIUM risk override")
        
        if len(cgm_data) >= 3:
            rise = cgm_data[-1] - cgm_data[-3]
            if rise > 40:
                if override_risk is None or override_risk == "LOW":
                    override_risk = "MEDIUM"
                rules_triggered.append(f"⚠️ CLINICAL RULE: Rapid rise of {rise:.0f} mg/dL detected - MEDIUM risk override")
        
        if override_risk is not None:
            model_risk = risk.get('risk_level', 'LOW')
            risk_priority = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 0}
            if risk_priority.get(override_risk, 0) > risk_priority.get(model_risk, 0):
                risk['risk_level'] = override_risk
                risk['requires_action'] = True
                risk['clinical_summary'] = f"[RULE OVERRIDE] {risk.get('clinical_summary', '')}"
        
        return risk, rules_triggered

# ========== ========== FIX: ENGINE ADAPTER WITH CORRECT SIGNATURE ========== ==========
class EngineAdapter:
    """Engine adapter that automatically detects and matches the correct signature"""
    
    def __init__(self):
        self.config = Config()
        self.orchestrator = ClinicalOrchestrator(self.config)
        self._signature_used = None
        self._model_version = "v1.0"
    
    def predict(self, cgm_data: List[float], carbs: int, protein: int, fat: int) -> Dict:
        """Universal adapter that tries multiple signature patterns"""
        
        # Prepare inputs
        cgm_array = np.array(cgm_data, dtype=np.float32)
        meal_context = {
            "carbs": carbs,
            "protein": protein,
            "fat": fat
        }
        top_k_value = 10
        
        # Get the actual function signature
        sig = inspect.signature(self.orchestrator.run)
        params = list(sig.parameters.keys())
        
        # Try different calling patterns based on parameter names
        try:
            # Pattern 1: Check if 'cgm' is a parameter
            if 'cgm' in params:
                result = self.orchestrator.run(
                    cgm=cgm_array,
                    meal_context=meal_context,
                    top_k=top_k_value
                )
                self._signature_used = "cgm_keyword"
                return result
            
            # Pattern 2: Check if 'cgm_data' is a parameter
            elif 'cgm_data' in params:
                result = self.orchestrator.run(
                    cgm_data=cgm_array,
                    meal_context=meal_context,
                    top_k=top_k_value
                )
                self._signature_used = "cgm_data_keyword"
                return result
            
            # Pattern 3: Check if 'data' is a parameter
            elif 'data' in params:
                result = self.orchestrator.run(
                    data=cgm_array,
                    context=meal_context,
                    k=top_k_value
                )
                self._signature_used = "data_keyword"
                return result
            
            # Pattern 4: Positional arguments (no parameter names)
            elif len(params) >= 2:
                # Try positional: run(cgm_array, meal_context, top_k)
                result = self.orchestrator.run(cgm_array, meal_context, top_k_value)
                self._signature_used = "positional"
                return result
            
            else:
                # Pattern 5: Single dict argument
                result = self.orchestrator.run({
                    "cgm": cgm_array,
                    "meal_context": meal_context,
                    "top_k": top_k_value
                })
                self._signature_used = "dict_input"
                return result
                
        except TypeError as e:
            # Log the actual signature for debugging
            st.error(f"Engine signature mismatch. Expected parameters: {params}")
            st.error(f"Error: {str(e)}")
            return {"error": f"engine_signature_mismatch: {str(e)}"}
        except Exception as e:
            st.error(f"Engine error: {str(e)}")
            return {"error": str(e)}
    
    def get_signature(self) -> str:
        return self._signature_used or "unknown"
    
    def get_model_version(self) -> str:
        return self._model_version

# ========== HELPER FUNCTIONS ==========

def compute_input_hash(cgm_data: List[float], carbs: int, protein: int, fat: int) -> str:
    if not cgm_data:
        return hashlib.md5("empty".encode()).hexdigest()
    cgm_str = ",".join([f"{x:.2f}" for x in cgm_data])
    input_str = f"{cgm_str}|{carbs}|{protein}|{fat}"
    return hashlib.md5(input_str.encode()).hexdigest()

def safe_last_value(data: List[float], default: float = 120.0) -> float:
    if data and len(data) > 0:
        try:
            return float(data[-1])
        except (ValueError, TypeError):
            return default
    return default

def generate_sample_cgm_fixed() -> List[float]:
    return [120, 122, 118, 121, 125, 128, 130, 135, 132, 129]

def validate_cgm_data(data: List[float], required_length: int = REQUIRED_READINGS) -> Tuple[bool, str]:
    if not data or len(data) == 0:
        return False, "No data provided"
    if len(data) != required_length:
        return False, f"Expected {required_length} readings, got {len(data)}"
    for i, x in enumerate(data):
        if x is None:
            return False, f"None value at position {i+1}"
        try:
            val = float(x)
            if np.isnan(val):
                return False, f"NaN at position {i+1}"
            if val < CGM_MIN or val > CGM_MAX:
                return False, f"Value {val:.1f} at {i+1} outside range ({CGM_MIN:.0f}-{CGM_MAX:.0f})"
        except (ValueError, TypeError):
            return False, f"Invalid value '{x}' at position {i+1}"
    return True, "Valid"

def validate_predictions(pred_means: List[float]) -> bool:
    if not pred_means:
        return False
    try:
        return all(
            isinstance(x, (int, float)) and 
            not np.isnan(x) and 
            CGM_MIN <= x <= CGM_MAX
            for x in pred_means
        )
    except:
        return False

def get_pred(preds: Dict, key: str, sub: str, default: float = 0.0) -> float:
    if not preds or not isinstance(preds, dict):
        return default
    pred_key = preds.get(key)
    if not isinstance(pred_key, dict):
        return default
    value = pred_key.get(sub)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def compute_velocity(cgm_data: List[float], model_velocity: Optional[float] = None) -> Tuple[float, float]:
    computed_velocity = 0.0
    if cgm_data and len(cgm_data) >= 3:
        try:
            time_diff = 2 * CGM_INTERVAL_MINUTES
            computed_velocity = (cgm_data[-1] - cgm_data[-3]) / time_diff
        except Exception:
            computed_velocity = 0.0
    
    if model_velocity is not None and abs(model_velocity) > 0.01:
        if abs(model_velocity) > 1.0:
            alpha = 0.4
        else:
            alpha = 0.6
        fused = alpha * computed_velocity + (1 - alpha) * model_velocity
        return fused, computed_velocity
    else:
        return computed_velocity, computed_velocity

def compute_trust_score(model_velocity: float, computed_velocity: float) -> Dict:
    if abs(model_velocity) < 0.01 and abs(computed_velocity) < 0.01:
        return {'score': 1.0, 'label': 'HIGH', 'message': 'Model agrees: stable glucose'}
    
    max_val = max(abs(model_velocity), abs(computed_velocity), 0.01)
    diff = abs(model_velocity - computed_velocity)
    agreement = max(0, 1 - min(diff / max_val, 1.0))
    
    if agreement > 0.8:
        return {'score': agreement, 'label': 'HIGH', 'message': 'Model and data agree'}
    elif agreement > 0.5:
        return {'score': agreement, 'label': 'MEDIUM', 'message': 'Moderate agreement'}
    else:
        return {'score': agreement, 'label': 'LOW', 'message': '⚠️ Model drift detected'}

def compute_reliability_score(preds: Dict, horizon: str = '30min') -> Dict:
    try:
        if not preds or not isinstance(preds, dict):
            return {'score': 0.0, 'label': 'INVALID', 'color': 'confidence-low'}
        
        pred_horizon = preds.get(horizon, {})
        lower = pred_horizon.get('lower', 0)
        upper = pred_horizon.get('upper', 0)
        
        if lower is None or upper is None:
            return {'score': 0.0, 'label': 'INVALID', 'color': 'confidence-low'}
        
        try:
            lower = float(lower)
            upper = float(upper)
        except (ValueError, TypeError):
            return {'score': 0.0, 'label': 'INVALID', 'color': 'confidence-low'}
        
        if lower >= upper:
            return {'score': 0.0, 'label': 'INVALID', 'color': 'confidence-low'}
        
        uncertainty = upper - lower
        
        if uncertainty < 10:
            return {'score': 1.0, 'label': 'HIGH', 'color': 'confidence-high'}
        elif uncertainty < 20:
            return {'score': 0.7, 'label': 'MEDIUM', 'color': 'confidence-medium'}
        else:
            return {'score': 0.3, 'label': 'LOW', 'color': 'confidence-low'}
    except:
        return {'score': 0.0, 'label': 'INVALID', 'color': 'confidence-low'}

def safe_ci_bounds(mean: float, lower: float, upper: float) -> Tuple[float, float]:
    lower = max(lower, CGM_MIN)
    upper = min(upper, CGM_MAX)
    
    if lower > mean:
        lower = mean - 1
    if upper < mean:
        upper = mean + 1
    
    return lower, upper

def detect_anomalies(cgm_data: List[float]) -> List[Dict]:
    anomalies = []
    if not cgm_data or len(cgm_data) < 3:
        return anomalies
    try:
        if np.std(cgm_data) < 1.0:
            anomalies.append({
                'type': 'FLAT_PATTERN',
                'severity': 'WARNING',
                'message': 'Flat glucose pattern detected — possible sensor issue'
            })
        diffs = np.diff(cgm_data)
        if np.max(np.abs(diffs)) > 30:
            anomalies.append({
                'type': 'RAPID_CHANGE',
                'severity': 'CAUTION',
                'message': f'Rapid glucose change (max delta: {np.max(np.abs(diffs)):.0f} mg/dL)'
            })
        if np.min(cgm_data) < 60:
            anomalies.append({
                'type': 'LOW_VALUE',
                'severity': 'ALERT',
                'message': f'Very low glucose: {np.min(cgm_data):.0f} mg/dL'
            })
        if np.max(cgm_data) > 200:
            anomalies.append({
                'type': 'HIGH_VALUE',
                'severity': 'ALERT',
                'message': f'Very high glucose: {np.max(cgm_data):.0f} mg/dL'
            })
    except:
        pass
    return anomalies

def get_current_glucose(risk: Dict, result: Dict, cgm_data: List[float]) -> float:
    if risk and risk.get('current') is not None:
        return float(risk['current'])
    if result and result.get('current_glucose') is not None:
        return float(result['current_glucose'])
    return safe_last_value(cgm_data)

def handle_unknown_risk(risk_level: str) -> None:
    if risk_level == "UNKNOWN":
        st.error("⚠️ MODEL UNCERTAINTY: Risk level is unknown - clinical interpretation unreliable")
        st.warning("Check engine output or data quality before making clinical decisions")

# ========== PLOTTING ==========
def plot_predictions(cgm_data: List[float], preds: Dict, current: float, 
                    preds_valid: bool = True, rules_triggered: List[str] = None):
    if not cgm_data or len(cgm_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=450)
        return fig
    
    fig = go.Figure()
    
    start = datetime.now() - timedelta(minutes=CGM_INTERVAL_MINUTES*(len(cgm_data)-1))
    times = [start + timedelta(minutes=CGM_INTERVAL_MINUTES*i) for i in range(len(cgm_data))]
    
    fig.add_trace(go.Scatter(
        x=times, y=cgm_data, mode='lines+markers', name='Historical CGM',
        line=dict(color='#4A90D9', width=2.5), marker=dict(size=8, color='#4A90D9')
    ))
    
    fig.add_trace(go.Scatter(
        x=[times[-1]], y=[current], mode='markers', name='Current',
        marker=dict(color='#FF6B6B', size=14, symbol='star')
    ))
    
    future_times = [
        times[-1] + timedelta(minutes=30),
        times[-1] + timedelta(minutes=60),
        times[-1] + timedelta(minutes=120)
    ]
    
    pred_means = [
        get_pred(preds, '30min', 'mean'),
        get_pred(preds, '60min', 'mean'),
        get_pred(preds, '120min', 'mean')
    ]
    
    if preds_valid and preds and validate_predictions(pred_means) and len(pred_means) == len(future_times):
        pred_lower = [
            get_pred(preds, '30min', 'lower'),
            get_pred(preds, '60min', 'lower'),
            get_pred(preds, '120min', 'lower')
        ]
        pred_upper = [
            get_pred(preds, '30min', 'upper'),
            get_pred(preds, '60min', 'upper'),
            get_pred(preds, '120min', 'upper')
        ]
        
        for i in range(len(pred_means)):
            pred_lower[i], pred_upper[i] = safe_ci_bounds(
                pred_means[i], pred_lower[i], pred_upper[i]
            )
        
        fig.add_trace(go.Scatter(
            x=[times[-1]] + future_times, y=[current] + pred_means,
            mode='lines+markers', name='Predicted',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            marker=dict(size=10, color='#FF6B6B')
        ))
        
        fig.add_trace(go.Scatter(
            x=[times[-1]] + future_times + future_times[::-1] + [times[-1]],
            y=[current] + pred_upper + pred_lower[::-1] + [current],
            fill='toself', fillcolor='rgba(255,107,107,0.15)',
            line=dict(color='rgba(255,107,107,0)'), name='95% CI'
        ))
    else:
        st.warning("⚠️ No valid predictions available - showing historical data only")
    
    fig.add_hrect(y0=70, y1=140, fillcolor='rgba(40,167,69,0.1)', line_width=0, name='Normal Range')
    fig.add_hrect(y0=0, y1=70, fillcolor='rgba(255,0,0,0.08)', line_width=0, name='Hypoglycemia')
    fig.add_hrect(y0=180, y1=300, fillcolor='rgba(255,165,0,0.08)', line_width=0, name='Hyperglycemia')
    
    if rules_triggered:
        annotation_text = "<br>".join(rules_triggered[:2])
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
    
    fig.update_layout(
        title='Glucose Trend with 30/60/120 min Predictions',
        xaxis_title='Time', yaxis_title='Glucose (mg/dL)',
        yaxis_range=[CGM_MIN, 300], hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450
    )
    return fig

# ========== INPUT RENDERERS ==========
def render_cgm_slider_input() -> List[float]:
    st.markdown("### 📊 Enter 10 CGM Readings")
    st.caption("Adjust each reading using the sliders below")
    cols = st.columns(5)
    cgm_data = []
    for i in range(10):
        col_idx = i % 5
        with cols[col_idx]:
            slider_key = f"cgm_slider_{i}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = 120.0
            val = st.slider(
                f"#{i+1}", min_value=CGM_MIN, max_value=CGM_MAX,
                value=st.session_state[slider_key], step=1.0,
                key=slider_key, format="%.0f"
            )
            cgm_data.append(float(val))
    return cgm_data

def render_cgm_text_input() -> Optional[List[float]]:
    st.markdown("### 📝 Enter 10 CGM Readings")
    st.caption("Enter exactly 10 comma-separated values (40-250 mg/dL)")
    default_values = generate_sample_cgm_fixed()
    default_str = ", ".join([str(int(v)) for v in default_values])
    cgm_input = st.text_area(
        "CGM readings (comma-separated)",
        value=default_str, height=60,
        help="Example: 120, 122, 118, 121, 125, 128, 130, 135, 132, 129"
    )
    try:
        cgm_data = [float(x.strip()) for x in cgm_input.split(",") if x.strip()]
        is_valid, msg = validate_cgm_data(cgm_data, REQUIRED_READINGS)
        if is_valid:
            st.success("✅ Valid: Exactly 10 readings")
            return cgm_data
        else:
            st.error(f"❌ {msg}")
            return None
    except Exception as e:
        st.error(f"❌ Invalid input: {str(e)}")
        return None

def render_cgm_quick_input() -> Optional[List[float]]:
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
            st.session_state.analyze = False
            st.rerun()
    with col2:
        if st.button("📈 Rising", use_container_width=True):
            st.session_state.cgm_data = presets["Rising"]
            st.session_state.cgm_input_source = "preset"
            st.session_state.analyze = False
            st.rerun()
    with col3:
        if st.button("📉 Falling", use_container_width=True):
            st.session_state.cgm_data = presets["Falling"]
            st.session_state.cgm_input_source = "preset"
            st.session_state.analyze = False
            st.rerun()
    with col4:
        if st.button("🎯 Spike", use_container_width=True):
            st.session_state.cgm_data = presets["Spike"]
            st.session_state.cgm_input_source = "preset"
            st.session_state.analyze = False
            st.rerun()
    return st.session_state.get('cgm_data')

# ========== CACHE LOADER ==========
@st.cache_resource
def load_engine():
    return EngineAdapter()

# ========== MAIN APP ==========
def main():
    st.title("🩺 GlucoseGuard CDSS")
    st.caption("Clinical Decision Support System for Diabetes Management")
    
    # Initialize session state
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
    if 'input_hash' not in st.session_state:
        st.session_state.input_hash = compute_input_hash(
            st.session_state.cgm_data, st.session_state.carbs,
            st.session_state.protein, st.session_state.fat
        )
    if 'prediction_audit' not in st.session_state:
        st.session_state.prediction_audit = []
    if 'risk_warnings' not in st.session_state:
        st.session_state.risk_warnings = []
    if 'reasoning_history' not in st.session_state:
        st.session_state.reasoning_history = []
    
    # Initialize components
    reasoning_engine = ClinicalReasoningEngine()
    audit = PredictionAudit()
    rule_engine = ClinicalRuleEngine()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        input_mode = st.radio(
            "Input Mode",
            ["🎛️ Sliders", "📝 Text Entry", "⚡ Quick Presets"],
            help="Sliders: Best UX | Text: Copy-paste | Presets: Quick testing"
        )
        st.divider()
        
        if input_mode == "🎛️ Sliders":
            cgm_data = render_cgm_slider_input()
            if cgm_data and len(cgm_data) == REQUIRED_READINGS:
                st.session_state.cgm_data = cgm_data
                st.session_state.cgm_input_source = "sliders"
        elif input_mode == "📝 Text Entry":
            cgm_data = render_cgm_text_input()
            if cgm_data and len(cgm_data) == REQUIRED_READINGS:
                st.session_state.cgm_data = cgm_data
                st.session_state.cgm_input_source = "text"
        else:
            render_cgm_quick_input()
        
        if st.session_state.cgm_data:
            st.divider()
            st.markdown("### 📊 Current Data")
            st.write(f"Readings: {len(st.session_state.cgm_data)}")
            st.line_chart(st.session_state.cgm_data, height=100)
            st.caption(f"Current: {safe_last_value(st.session_state.cgm_data):.0f} mg/dL")
        
        st.divider()
        st.markdown("### 🍽️ Meal Context")
        carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=st.session_state.carbs, step=5)
        protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=st.session_state.protein, step=5)
        fat = st.number_input("Fat (g)", min_value=0, max_value=100, value=st.session_state.fat, step=5)
        
        current_hash = compute_input_hash(st.session_state.cgm_data, carbs, protein, fat)
        if current_hash != st.session_state.input_hash:
            st.session_state.input_hash = current_hash
            st.session_state.analyze = False
        
        st.session_state.carbs = carbs
        st.session_state.protein = protein
        st.session_state.fat = fat
        
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            cgm_data = st.session_state.cgm_data
            is_valid, msg = validate_cgm_data(cgm_data, REQUIRED_READINGS)
            if is_valid:
                st.session_state.analyze = True
            else:
                st.error(f"❌ {msg}")
    
    if not st.session_state.get('analyze', False):
        st.info("👈 Select input mode and enter 10 CGM readings, then click 'Analyze'")
        if st.session_state.cgm_data:
            with st.expander("📊 Current CGM Data Preview", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.cgm_data, mode='lines+markers',
                    name='CGM', marker=dict(size=10)
                ))
                fig.update_layout(height=250, title="CGM Readings")
                st.plotly_chart(fig, use_container_width=True)
        return
    
    # ========== ANALYZE ==========
    try:
        cgm_data = st.session_state.cgm_data
        is_valid, msg = validate_cgm_data(cgm_data, REQUIRED_READINGS)
        if not is_valid:
            st.error(f"❌ {msg}")
            st.session_state.analyze = False
            st.stop()
        
        anomalies = detect_anomalies(cgm_data)
        if anomalies:
            with st.expander("⚠️ Data Anomalies Detected", expanded=False):
                for anomaly in anomalies:
                    severity_color = "🔴" if anomaly['severity'] == 'ALERT' else "🟡"
                    st.warning(f"{severity_color} {anomaly['message']}")
        
        engine = load_engine()
        
        with st.spinner("Analyzing glucose data..."):
            result = engine.predict(
                cgm_data,
                st.session_state.carbs,
                st.session_state.protein,
                st.session_state.fat
            )
            
            if result.get("error"):
                st.error(f"Engine error: {result['error']}")
                st.session_state.analyze = False
                st.stop()
            
            is_valid, msg = validate_engine_output(result)
            if not is_valid:
                st.error(f"Invalid engine output: {msg}")
                st.session_state.analyze = False
                st.stop()
        
        # Normalize components
        risk = RiskNormalizer.normalize(result.get('risk') or {})
        preds_raw, preds_valid = PredictionNormalizer.normalize(result.get('predictions'))
        
        # Extract all required variables
        food_recs = ClinicalGuard.normalize_food_recs(result.get('food_recommendations'))
        activity = ClinicalGuard.normalize_activity(result.get('activity'))
        meal_plan = ClinicalGuard.normalize_meal_plan(result.get('meal_plan'))
        
        if not preds_valid or preds_raw is None:
            st.warning("⚠️ Predictions had invalid values - using defaults")
            preds = {
                '30min': {'mean': 0.0, 'lower': 0.0, 'upper': 0.0},
                '60min': {'mean': 0.0, 'lower': 0.0, 'upper': 0.0},
                '120min': {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
            }
            preds_valid = False
        else:
            preds = preds_raw
        
        # Velocity fusion
        model_velocity = risk.get('glucose_velocity')
        velocity, computed_velocity = compute_velocity(cgm_data, model_velocity)
        risk['glucose_velocity'] = velocity
        
        current = get_current_glucose(risk, result, cgm_data)
        
        # Apply clinical rules
        risk, rules_triggered = rule_engine.apply_rules(cgm_data, preds, risk)
        
        # Get updated risk level
        risk_level = risk.get('risk_level', 'LOW')
        
        if rules_triggered:
            with st.expander("⚕️ Clinical Rules Triggered", expanded=True):
                for rule in rules_triggered:
                    st.warning(rule)
        
        # Trust score
        trust = compute_trust_score(velocity, computed_velocity)
        
        # Reliability
        reliability = compute_reliability_score(preds, '30min')
        
        # Reasoning
        reasoning = reasoning_engine.generate_reasoning(
            risk, preds, current, reliability, velocity, trust, rules_triggered
        )
        st.session_state.reasoning_history = reasoning['history']
        
        # Audit
        audit.log_prediction(
            cgm_data, preds, risk, velocity, reliability, trust,
            engine.get_signature(), engine.get_model_version(),
            rules_triggered
        )
        
        # Unknown risk handling
        if risk_level == "UNKNOWN":
            handle_unknown_risk(risk_level)
        
        # Risk banner
        risk_colors = {
            'HIGH': 'risk-high', 'MEDIUM': 'risk-medium',
            'LOW': 'risk-low', 'UNKNOWN': 'risk-unknown'
        }
        if rules_triggered and risk_level in ['HIGH', 'MEDIUM']:
            color_style = 'risk-override'
        else:
            color_style = risk_colors.get(risk_level, 'risk-low')
        
        dominant_risk = risk.get('dominant_risk', 'NONE')
        clinical_summary = risk.get('clinical_summary', 'No summary available')
        
        st.markdown(f"""
            <div class="{color_style}">
                🚨 RISK: {risk_level} — {dominant_risk}
                <br><small>{clinical_summary}</small>
                {f'<br><small>⚕️ {len(rules_triggered)} clinical rules triggered</small>' if rules_triggered else ''}
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        pred_30 = get_pred(preds, '30min', 'mean')
        pred_60 = get_pred(preds, '60min', 'mean')
        pred_120 = get_pred(preds, '120min', 'mean')
        
        with col1:
            st.metric("Current", f"{current:.0f} mg/dL")
        with col2:
            st.metric("30-min", f"{pred_30:.0f}", delta=f"{pred_30 - current:+.0f}")
        with col3:
            st.metric("60-min", f"{pred_60:.0f}", delta=f"{pred_60 - current:+.0f}")
        with col4:
            st.metric("120-min", f"{pred_120:.0f}", delta=f"{pred_120 - current:+.0f}")
        with col5:
            st.metric("Velocity", f"{velocity:.2f} mg/dL/min")
        
        # Reasoning with uncertainty
        st.info(f"🧠 {reasoning['summary']}")
        
        # Show confidence, trust, and rules
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div style="padding: 5px;">
                    <span class="{reliability['color']}">
                        {reliability['label']} Confidence ({reliability['score']:.0%})
                    </span>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style="padding: 5px;">
                    <span class="confidence-{trust['label'].lower()}">
                        Model Trust: {trust['label']} ({trust['score']:.0%})
                    </span>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            if rules_triggered:
                st.markdown(f"""
                    <div style="padding: 5px; background-color: #d4edda; border-radius: 4px;">
                        ⚕️ {len(rules_triggered)} Clinical Rules
                    </div>
                """, unsafe_allow_html=True)
        
        # Full reasoning
        with st.expander("📋 Full Clinical Reasoning", expanded=False):
            st.markdown(reasoning['full_reasoning'])
            if reasoning['actionable']:
                st.warning("⚠️ Actionable insights detected - clinical attention advised")
            if trust['label'] == 'LOW':
                st.warning(f"⚠️ {trust['message']}")
        
        # Plot
        st.plotly_chart(plot_predictions(cgm_data, preds, current, preds_valid, rules_triggered), use_container_width=True)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Glucose Predictions",
            "🍽️ Food Recommendations",
            "🏃 Activity Plan",
            "📋 Clinical Summary",
            "📜 Audit Trail"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 Prediction Details")
                pred_df = pd.DataFrame({
                    'Horizon': ['30 min', '60 min', '120 min'],
                    'Mean': [get_pred(preds, '30min', 'mean'),
                            get_pred(preds, '60min', 'mean'),
                            get_pred(preds, '120min', 'mean')],
                    'Lower (95% CI)': [get_pred(preds, '30min', 'lower'),
                                      get_pred(preds, '60min', 'lower'),
                                      get_pred(preds, '120min', 'lower')],
                    'Upper (95% CI)': [get_pred(preds, '30min', 'upper'),
                                      get_pred(preds, '60min', 'upper'),
                                      get_pred(preds, '120min', 'upper')]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                if not preds_valid:
                    st.warning("⚠️ Some predictions were invalid and defaulted to 0")
            with col2:
                st.markdown("### 🎯 Risk Metrics")
                st.metric("Peak", f"{risk.get('peak', 0):.0f} mg/dL")
                st.metric("Trough", f"{risk.get('trough', 0):.0f} mg/dL")
                st.metric("Risk Score", f"{risk.get('risk_score', 0):.2f}")
                st.metric("Requires Action", "✅ Yes" if risk.get('requires_action', False) else "❌ No")
                st.metric("Engine Signature", engine.get_signature())
                st.metric("Model Confidence", reliability['label'])
                st.metric("Model Trust", trust['label'])
                st.metric("Predictions Valid", "✅" if preds_valid else "⚠️ Partial")
                if rules_triggered:
                    st.metric("Clinical Rules", len(rules_triggered))
                
                st.markdown("### 📊 Input Info")
                st.write(f"- **Mode:** {st.session_state.cgm_input_source}")
                st.write(f"- **Readings:** {len(cgm_data)}")
                if cgm_data:
                    try:
                        clean_data = [x for x in cgm_data if x is not None and not np.isnan(x)]
                        if clean_data:
                            st.write(f"- **Range:** {np.min(clean_data):.0f} - {np.max(clean_data):.0f} mg/dL")
                    except:
                        pass
        
        with tab2:
            if food_recs is not None and not food_recs.empty:
                st.markdown("### 🍽️ Top Food Recommendations")
                if meal_plan:
                    for meal, items in meal_plan.items():
                        if items:
                            with st.expander(f"🥗 {meal}", expanded=True):
                                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                with st.expander("📋 All Recommendations", expanded=False):
                    st.dataframe(food_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("No food recommendations available")
        
        with tab3:
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
                st.write(f"- **Glucose Velocity:** {velocity:.2f} mg/dL/min")
                st.write(f"- **Computed Velocity:** {computed_velocity:.2f} mg/dL/min")
                st.write(f"- **Trend Strength:** {risk.get('trend_strength', 0):.2f}")
                st.write(f"- **Model Confidence:** {reliability['label']}")
                st.write(f"- **Model Trust:** {trust['label']} ({trust['score']:.0%})")
                st.write(f"- **Predictions Valid:** {'✅' if preds_valid else '⚠️ Partial'}")
                if rules_triggered:
                    st.write(f"- **Clinical Rules:** {len(rules_triggered)} triggered")
            with cols[1]:
                st.markdown("#### Predicted Events")
                st.write(f"- **Predicted Peak:** {risk.get('peak', 0):.0f} mg/dL")
                st.write(f"- **Predicted Trough:** {risk.get('trough', 0):.0f} mg/dL")
                st.write(f"- **Spike Magnitude:** {risk.get('upward_spike', 0):.0f} mg/dL")
                st.write(f"- **Drop Magnitude:** {risk.get('downward_drop', 0):.0f} mg/dL")
            
            st.markdown("#### Clinical Interpretation")
            st.info(clinical_summary)
            st.markdown("#### Detailed Reasoning")
            st.caption(reasoning['full_reasoning'])
            
            if rules_triggered:
                st.markdown("#### Clinical Rules Triggered")
                for rule in rules_triggered:
                    st.warning(rule)
            
            st.markdown("#### Meal Context")
            st.write(f"- **Carbs:** {st.session_state.carbs}g")
            st.write(f"- **Protein:** {st.session_state.protein}g")
            st.write(f"- **Fat:** {st.session_state.fat}g")
            
            if st.button("📥 Download Audit Trail (JSON)"):
                audit_json = audit.export_audit()
                st.download_button(
                    label="Download JSON",
                    data=audit_json,
                    file_name=f"cdss_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with tab5:
            st.markdown("### 📜 Prediction Audit Trail")
            audit_history = st.session_state.prediction_audit
            if audit_history:
                st.info(f"Total predictions logged: {len(audit_history)}")
                latest = audit_history[-1]
                st.markdown(f"**Latest Prediction:** {latest['timestamp']}")
                st.markdown(f"**Model Version:** {latest.get('model_version', 'unknown')}")
                st.markdown(f"**Engine Signature:** {latest.get('engine_signature', 'unknown')}")
                if latest.get('clinical_rules'):
                    st.markdown(f"**Clinical Rules Triggered:** {len(latest['clinical_rules'])}")
                
                st.markdown("#### Prediction History")
                history_df = pd.DataFrame([
                    {
                        'Time': entry['timestamp'],
                        'Risk': entry['risk'].get('risk_level', 'UNKNOWN'),
                        'Velocity': entry.get('velocity', 0),
                        'Reliability': entry.get('reliability', {}).get('label', 'UNKNOWN'),
                        'Trust': entry.get('trust', {}).get('label', 'UNKNOWN'),
                        'Rules': len(entry.get('clinical_rules', []))
                    }
                    for entry in audit_history[-10:]
                ])
                st.dataframe(history_df, use_container_width=True)
                
                if st.session_state.reasoning_history:
                    st.markdown("#### Reasoning History")
                    reasoning_df = pd.DataFrame([
                        {
                            'Time': entry['timestamp'],
                            'Actionable': entry['actionable'],
                            'Risk': entry['risk_level'],
                            'Rules': len(entry.get('rule_overrides', [])),
                            'Reasoning': entry['reasoning'][:100] + '...'
                        }
                        for entry in st.session_state.reasoning_history[-5:]
                    ])
                    st.dataframe(reasoning_df, use_container_width=True)
                
                st.markdown("#### Export Audit")
                if st.button("📥 Export Full Audit (JSON)"):
                    audit_json = audit.export_audit()
                    st.download_button(
                        label="Download Full Audit",
                        data=audit_json,
                        file_name=f"cdss_full_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("No predictions logged yet")
        
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analyze = False
            st.session_state.cgm_data = generate_sample_cgm_fixed()
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        if st.checkbox("Show detailed error (debug mode)", value=False):
            st.exception(e)
        st.session_state.analyze = False
        st.stop()

if __name__ == "__main__":
    main()
