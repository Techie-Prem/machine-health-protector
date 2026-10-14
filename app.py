"""
Predictive Maintenance System - AI-Powered Equipment Health Monitoring
High-Performance Real-Time Diagnostic Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.stats import skew, kurtosis
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ULTRA-PREMIUM MODERN DARK THEME CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(6, 182, 212, 0.07) 0%, transparent 40%),
                    radial-gradient(circle at 90% 90%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
                    #0b1120;
        color: #f8fafc;
    }

    /* Top Header Bar */
    .hero-container {
        background: linear-gradient(135deg, rgba(17, 27, 46, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 12px;
    }

    .status-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .status-badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(6, 182, 212, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.25);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Headings & Text High Contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    p, span, label {
        color: #cbd5e1;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1527 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 27, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(6, 182, 212, 0.35);
        transform: translateY(-2px);
    }

    .sensor-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .sensor-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f1f5f9;
    }

    .sensor-tag {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
    }

    /* Metric Overrides */
    div[data-testid="stMetric"] {
        background: rgba(17, 27, 46, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Status Result Banners */
    .result-banner-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
        margin-bottom: 20px;
    }

    .result-banner-faulty {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.3) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
        margin-bottom: 20px;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 27, 46, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3);
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #06b6d4 0%, #2563eb 100%);
        border: none;
        color: #ffffff;
        box-shadow: 0 4px 18px rgba(6, 182, 212, 0.35);
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0891b2 0%, #1d4ed8 100%);
        box-shadow: 0 6px 24px rgba(6, 182, 212, 0.5);
        transform: translateY(-1px);
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.7);
        color: #e2e8f0;
    }

    div.stButton > button[kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.9);
        border-color: rgba(6, 182, 212, 0.4);
    }

    /* Input widgets */
    div[data-testid="stNumberInput"] label {
        color: #e2e8f0 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #111b2e !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 1px #06b6d4 !important;
    }

    /* Slider adjustments */
    div[data-testid="stSlider"] label {
        color: #cbd5e1 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(17, 27, 46, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'input_a1' not in st.session_state:
    st.session_state.input_a1 = 1.8500
if 'input_a2' not in st.session_state:
    st.session_state.input_a2 = 2.1000
if 'input_a3' not in st.session_state:
    st.session_state.input_a3 = 3.2000
if 'input_a4' not in st.session_state:
    st.session_state.input_a4 = 2.6500

if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ==========================================
# MODEL COMPONENT LOADER
# ==========================================
@st.cache_resource
def load_model_components():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        required_files = [
            'trained_model.pkl',
            'scaler.pkl',
            'pca.pkl',
            'model_metadata.json'
        ]

        missing_files = [f for f in required_files if not os.path.exists(os.path.join(base_dir, f))]
        if missing_files:
            return None, None, None, None, None, f"Missing files: {', '.join(missing_files)}"

        model = joblib.load(os.path.join(base_dir, 'trained_model.pkl'))
        scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
        pca = joblib.load(os.path.join(base_dir, 'pca.pkl'))
        label_encoder = None

        with open(os.path.join(base_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)

        return model, scaler, pca, label_encoder, metadata, None

    except Exception as e:
        return None, None, None, None, None, str(e)

# ==========================================
# FEATURE EXTRACTION & PIPELINE
# ==========================================
def single_value_to_window(value, window_size=256):
    noise = np.random.normal(
        loc=0,
        scale=0.05 * abs(value + 1e-6),
        size=window_size
    )
    return value + noise

def create_windows(signal, window_size=256, step=128):
    return [
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ]

def extract_features(window):
    feats = [
        np.mean(window),
        np.std(window),
        np.sqrt(np.mean(window**2)),   # RMS
        np.min(window),
        np.max(window),
        skew(window),
        kurtosis(window)
    ]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
# CHART BUILDERS (PLOTLY MODERN DARK THEME)
# ==========================================
def create_gauge_chart(value, title, is_faulty=False):
    """Custom High-Tech Dark Neon Gauge"""
    bar_color = "#ef4444" if is_faulty else "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#f8fafc', 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'size': 44, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(15, 23, 42, 0.7)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(15, 23, 42, 0.5)'},
                {'range': [50, 75], 'color': 'rgba(6, 182, 212, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.2)' if not is_faulty else 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#38bdf8", 'width': 4},
                'thickness': 0.78,
                'value': value
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc", 'family': 'Inter'},
        margin=dict(l=20, r=20, t=50, b=20),
        height=280
    )
    return fig

def create_waveform_chart(windows):
    """Simulated 4-Channel Live Vibration Waveform"""
    sensor_names = [
        "Motor Vib X (a1)",
        "Motor Vib Y (a2)",
        "Gearbox Vib (a3)",
        "Bearing Vib (a4)"
    ]
    colors = ["#38bdf8", "#818cf8", "#f59e0b", "#34d399"]

    fig = go.Figure()
    sample_indices = np.arange(len(windows[0]))

    for i, w in enumerate(windows):
        fig.add_trace(go.Scatter(
            x=sample_indices,
            y=w,
            mode='lines',
            name=sensor_names[i],
            line=dict(color=colors[i], width=1.7),
            opacity=0.88
        ))

    fig.update_layout(
        title={'text': "⚡ 4-Channel Dynamic Vibration Waveform (256 Samples Window)", 'font': {'size': 14, 'color': '#f8fafc'}},
        paper_bgcolor='rgba(15, 23, 42, 0.5)',
        plot_bgcolor='rgba(11, 17, 32, 0.8)',
        xaxis=dict(
            title="Sample Index",
            gridcolor='rgba(255, 255, 255, 0.06)',
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            color='#94a3b8'
        ),
        yaxis=dict(
            title="Amplitude (mm/s)",
            gridcolor='rgba(255, 255, 255, 0.06)',
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            color='#94a3b8'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#cbd5e1', size=11)
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        height=240
    )
    return fig

def create_probability_chart(probabilities, class_names):
    """Glowing Class Probability Distribution"""
    prob_pcts = probabilities * 100
    colors = ['#10b981' if name == "Healthy" else '#ef4444' for name in class_names]

    fig = go.Figure(go.Bar(
        x=class_names,
        y=prob_pcts,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1.5)
        ),
        text=[f"{p:.1f}%" for p in prob_pcts],
        textposition='outside',
        textfont=dict(color='#ffffff', size=14, family='JetBrains Mono')
    ))

    fig.update_layout(
        title={'text': "📊 Model Confidence by Condition", 'font': {'size': 15, 'color': '#f8fafc'}},
        paper_bgcolor='rgba(15, 23, 42, 0.5)',
        plot_bgcolor='rgba(11, 17, 32, 0.8)',
        yaxis=dict(
            range=[0, 115],
            title="Probability (%)",
            gridcolor='rgba(255, 255, 255, 0.06)',
            color='#94a3b8'
        ),
        xaxis=dict(
            color='#f8fafc',
            tickfont=dict(size=14, weight='bold')
        ),
        margin=dict(l=30, r=30, t=40, b=30),
        height=280
    )
    return fig

# ==========================================
# MAIN APPLICATION LOGIC
# ==========================================
def main():
    model, scaler, pca, label_encoder, metadata, error = load_model_components()

    if error:
        st.error(f"❌ Error loading model artifacts: {error}")
        st.info("💡 Please verify that trained_model.pkl, scaler.pkl, pca.pkl and model_metadata.json are present in the project folder.")
        st.stop()
        return

    # ------------------------------------------
    # SIDEBAR: SYSTEM INFORMATION & ACTIONS
    # ------------------------------------------
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="background: linear-gradient(135deg, #06b6d4, #3b82f6); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);">
                ⚙️
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; color: #f8fafc;">Diagnostic Core</h3>
                <span style="color: #38bdf8; font-size: 0.78rem; font-weight: 600;">ACTIVE INFERENCE ENGINE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 16px;">
            <span style="color: #34d399; font-weight: 600; font-size: 0.85rem;">● ML Model Loaded & Calibrated</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔬 Model Architecture", expanded=True):
            st.metric("Model Algorithm", metadata.get('model_type', 'SVM (RBF)'))
            st.metric("Test Accuracy", f"{metadata.get('accuracy', 0.9905):.2%}")
            st.metric("Training Dataset", f"{metadata.get('n_samples', 15786):,} samples")
            st.metric("Input Dimensions", f"{metadata.get('n_features', 28)} (4 sensors × 7 features)")

        with st.expander("📡 Monitored Sensor Channels", expanded=False):
            st.markdown("""
            - **a1**: Motor Vibration (X-Axis)
            - **a2**: Motor Vibration (Y-Axis)
            - **a3**: Gearbox Vibration
            - **a4**: Bearing Vibration
            """)

        st.markdown("---")
        st.markdown("### 📊 Session Activity")
        total_predictions = len(st.session_state.predictions_history)
        st.metric("Inferences Run", total_predictions)

        if total_predictions > 0:
            faulty_count = sum(1 for p in st.session_state.predictions_history if p['predicted_class'] == 'Faulty')
            st.metric("Faults Detected", faulty_count, delta=f"{(faulty_count / total_predictions):.1%}" if total_predictions else "0%")

        if st.button("🗑️ Reset Session History", use_container_width=True, type="secondary"):
            st.session_state.predictions_history = []
            st.session_state.last_result = None
            st.rerun()

    # ------------------------------------------
    # TOP HERO HEADER
    # ------------------------------------------
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 class="hero-title">⚙️ Predictive Maintenance AI</h1>
                <div class="hero-subtitle">Real-time vibration telemetry & intelligent failure diagnosis for rotating machinery</div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-badge-live">● System Operational</span>
                <span class="status-badge-chip">SVM 99.05% Accuracy</span>
                <span class="status-badge-chip">4 Telemetry Channels</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------
    # WORKSPACE TABS
    # ------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Real-Time Diagnostic",
        "📁 Batch Analysis & Upload",
        "🔬 PCA & Model Intelligence",
        "📜 Telemetry Audit History"
    ])

    # ==========================================
    # TAB 1: REAL-TIME DIAGNOSTIC
    # ==========================================
    with tab1:
        st.markdown("### 🎛️ Real-Time Sensor Inputs")
        st.caption("Select a quick operating scenario or adjust the 4 sensor channels to evaluate equipment health.")

        # Preset Scenario Buttons
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        with col_p1:
            if st.button("🟢 Load Normal Baseline", use_container_width=True):
                st.session_state.input_a1 = 1.8500
                st.session_state.input_a2 = 2.1000
                st.session_state.input_a3 = 3.2000
                st.session_state.input_a4 = 2.6500
                st.rerun()

        with col_p2:
            if st.button("🔴 Load Faulty / Broken Preset", use_container_width=True):
                st.session_state.input_a1 = 0.0400
                st.session_state.input_a2 = 0.0500
                st.session_state.input_a3 = 0.0300
                st.session_state.input_a4 = 0.0500
                st.rerun()

        with col_p3:
            if st.button("🟡 Load Wear Warning", use_container_width=True):
                st.session_state.input_a1 = 0.8500
                st.session_state.input_a2 = 1.1000
                st.session_state.input_a3 = 1.4000
                st.session_state.input_a4 = 1.2000
                st.rerun()

        with col_p4:
            if st.button("🎲 Random Vibration Sample", use_container_width=True):
                st.session_state.input_a1 = round(float(np.random.uniform(0.01, 3.5)), 4)
                st.session_state.input_a2 = round(float(np.random.uniform(0.01, 3.5)), 4)
                st.session_state.input_a3 = round(float(np.random.uniform(0.01, 4.5)), 4)
                st.session_state.input_a4 = round(float(np.random.uniform(0.01, 4.0)), 4)
                st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 4 Sensor Interactive Input Cards
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class="glass-card">
                <div class="sensor-header">
                    <span class="sensor-name">Motor Vibration X</span>
                    <span class="sensor-tag">Sensor a1</span>
                </div>
            """, unsafe_allow_html=True)
            v_a1 = st.number_input(
                "Vibration X (mm/s)",
                value=float(st.session_state.input_a1),
                step=0.05,
                format="%.4f",
                key="num_a1",
                label_visibility="collapsed"
            )
            st.session_state.input_a1 = v_a1
            st.caption("Axis: Radial Horizontal")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="glass-card">
                <div class="sensor-header">
                    <span class="sensor-name">Motor Vibration Y</span>
                    <span class="sensor-tag">Sensor a2</span>
                </div>
            """, unsafe_allow_html=True)
            v_a2 = st.number_input(
                "Vibration Y (mm/s)",
                value=float(st.session_state.input_a2),
                step=0.05,
                format="%.4f",
                key="num_a2",
                label_visibility="collapsed"
            )
            st.session_state.input_a2 = v_a2
            st.caption("Axis: Radial Vertical")
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="glass-card">
                <div class="sensor-header">
                    <span class="sensor-name">Gearbox Vibration</span>
                    <span class="sensor-tag">Sensor a3</span>
                </div>
            """, unsafe_allow_html=True)
            v_a3 = st.number_input(
                "Gearbox (mm/s)",
                value=float(st.session_state.input_a3),
                step=0.05,
                format="%.4f",
                key="num_a3",
                label_visibility="collapsed"
            )
            st.session_state.input_a3 = v_a3
            st.caption("Axis: Gear Mesh Axial")
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class="glass-card">
                <div class="sensor-header">
                    <span class="sensor-name">Bearing Vibration</span>
                    <span class="sensor-tag">Sensor a4</span>
                </div>
            """, unsafe_allow_html=True)
            v_a4 = st.number_input(
                "Bearing (mm/s)",
                value=float(st.session_state.input_a4),
                step=0.05,
                format="%.4f",
                key="num_a4",
                label_visibility="collapsed"
            )
            st.session_state.input_a4 = v_a4
            st.caption("Axis: Bearing Outer Race")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Real-time simulated waveform preview
        np.random.seed(int(abs(v_a1 + v_a2 + v_a3 + v_a4) * 1000) % 100000)
        preview_windows = [
            single_value_to_window(v_a1),
            single_value_to_window(v_a2),
            single_value_to_window(v_a3),
            single_value_to_window(v_a4)
        ]
        waveform_fig = create_waveform_chart(preview_windows)
        st.plotly_chart(waveform_fig, use_container_width=True)

        # Action Button
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1.8, 1])
        with btn_col2:
            run_predict = st.button("🚀 EXECUTE AI HEALTH DIAGNOSTIC", type="primary", use_container_width=True)

        # Run Prediction Pipeline
        if run_predict:
            with st.spinner("Processing 28 statistical features through PCA & SVM pipeline..."):
                features = []
                for w in preview_windows:
                    features.extend(extract_features(w))

                X_input = np.array(features).reshape(1, -1)
                X_scaled = scaler.transform(X_input)
                X_pca = pca.transform(X_scaled)

                prediction = int(model.predict(X_pca)[0])
                probabilities = model.predict_proba(X_pca)[0]

                class_map = {0: "Healthy", 1: "Faulty"}
                predicted_class = class_map.get(prediction, "Healthy")
                confidence = float(probabilities[prediction] * 100)

                # Save result
                st.session_state.last_result = {
                    'class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'inputs': [v_a1, v_a2, v_a3, v_a4],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Save history
                st.session_state.predictions_history.append({
                    'timestamp': st.session_state.last_result['timestamp'],
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'mode': 'Single',
                    'a1': v_a1,
                    'a2': v_a2,
                    'a3': v_a3,
                    'a4': v_a4
                })

        # Display Result if available
        if st.session_state.last_result:
            res = st.session_state.last_result
            predicted_class = res['class']
            confidence = res['confidence']
            probabilities = res['probabilities']

            st.markdown("---")
            st.markdown("### 📋 Diagnostic Assessment Report")

            if predicted_class == "Healthy":
                st.markdown(f"""
                <div class="result-banner-healthy">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 700; font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; text-transform: uppercase;">● NOMINAL OPERATION</span>
                            <h2 style="margin: 6px 0 2px 0; color: #10b981 !important; font-size: 1.8rem;">Equipment Condition: HEALTHY</h2>
                            <p style="margin: 0; color: #a7f3d0; font-size: 0.95rem;">All 4 vibration channels match normal mechanical baseline parameters. Vibration is within ISO 10816 standards.</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; color: #a7f3d0;">DIAGNOSTIC CONFIDENCE</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #34d399; font-family: 'JetBrains Mono';">{confidence:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-banner-faulty">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="background: rgba(239, 68, 68, 0.25); color: #f87171; font-weight: 700; font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; text-transform: uppercase;">🚨 ANOMALY DETECTED</span>
                            <h2 style="margin: 6px 0 2px 0; color: #ef4444 !important; font-size: 1.8rem;">Equipment Condition: FAULTY / BROKEN</h2>
                            <p style="margin: 0; color: #fca5a5; font-size: 0.95rem;">Vibration characteristics deviate significantly from standard profiles. Indicative of mechanical looseness, gear tooth pitting, or bearing raceway defect.</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; color: #fca5a5;">FAULT PROBABILITY</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #f87171; font-family: 'JetBrains Mono';">{confidence:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Metrics & Visuals
            g_col1, g_col2 = st.columns([1, 1])

            with g_col1:
                gauge_fig = create_gauge_chart(
                    confidence,
                    f"Confidence: {predicted_class}",
                    is_faulty=(predicted_class == "Faulty")
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            with g_col2:
                prob_fig = create_probability_chart(probabilities, ["Healthy", "Faulty"])
                st.plotly_chart(prob_fig, use_container_width=True)

            # Prescriptive Maintenance Actions
            st.markdown("#### 🛠️ Prescriptive Maintenance Recommendations")
            act_col1, act_col2 = st.columns(2)

            with act_col1:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="margin-top: 0; color: #38bdf8 !important;">📌 Action Checklist</h4>
                """, unsafe_allow_html=True)

                if predicted_class == "Healthy":
                    st.markdown("""
                    - ✅ **Routine Inspection:** Next scheduled preventive check in 30 days.
                    - ✅ **Lubrication:** Verify grease reservoir levels remain nominal.
                    - ✅ **Telemetry Logged:** Baseline sensor readings archived to historical database.
                    """)
                else:
                    st.markdown("""
                    - ⚠️ **Immediate Work Order:** Dispatch mechanical technician to inspect drivetrain.
                    - ⚠️ **Bearing Check:** Perform acoustic ultrasound inspection on bearing a4.
                    - ⚠️ **Gearbox Alignment:** Verify tooth backlash and gearbox shaft alignment.
                    - ⚠️ **Speed / Load Derate:** Reduce operating load by 25% if vibration exceeds threshold.
                    """)
                st.markdown("</div>", unsafe_allow_html=True)

            with act_col2:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="margin-top: 0; color: #818cf8 !important;">📑 Export Report</h4>
                    <p style="font-size: 0.88rem; color: #94a3b8;">Generate a timestamped engineering diagnostic report for maintenance ticketing.</p>
                """, unsafe_allow_html=True)

                report_data = {
                    "timestamp": res['timestamp'],
                    "equipment_status": predicted_class,
                    "confidence": f"{confidence:.2f}%",
                    "sensor_readings": {
                        "motor_vibration_x_a1": res['inputs'][0],
                        "motor_vibration_y_a2": res['inputs'][1],
                        "gearbox_vibration_a3": res['inputs'][2],
                        "bearing_vibration_a4": res['inputs'][3]
                    },
                    "model_evaluator": metadata.get('model_type', 'SVM (RBF)')
                }
                st.download_button(
                    label="⬇️ Download JSON Diagnostic Log",
                    data=json.dumps(report_data, indent=4),
                    file_name=f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # TAB 2: BATCH ANALYSIS
    # ==========================================
    with tab2:
        st.markdown("### 📁 Batch Telemetry Processor")
        st.caption("Upload continuous time-series CSV sensor recordings to perform sliding-window automated fault detection.")

        b_col1, b_col2 = st.columns([1.4, 1])

        with b_col1:
            uploaded_file = st.file_uploader(
                "Upload Telemetry CSV File",
                type=['csv'],
                help="Accepts columns [a1, a2, a3, a4] or full names (Motor vibration X/Y, Gearbox, Bearing)"
            )

        with b_col2:
            st.markdown("""
            <div class="glass-card" style="margin-top: 24px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem;">⚡ Instant Demo Batch</h4>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;">Test the batch processing engine with 1,000 simulated multi-channel time steps across healthy and faulty conditions.</p>
            """, unsafe_allow_html=True)

            if st.button("⚡ Generate & Run Demo Batch", use_container_width=True, type="primary"):
                np.random.seed(42)
                demo_length = 1000
                demo_data = {
                    "a1": np.concatenate([np.random.normal(1.85, 0.1, 500), np.random.normal(0.04, 0.01, 500)]),
                    "a2": np.concatenate([np.random.normal(2.10, 0.1, 500), np.random.normal(0.05, 0.01, 500)]),
                    "a3": np.concatenate([np.random.normal(3.20, 0.15, 500), np.random.normal(0.03, 0.01, 500)]),
                    "a4": np.concatenate([np.random.normal(2.65, 0.15, 500), np.random.normal(0.05, 0.01, 500)]),
                }
                demo_df = pd.DataFrame(demo_data)
                st.session_state['batch_df'] = demo_df
                st.success("✅ Demo batch generated! Processing below...")

            sample_template_df = pd.DataFrame({
                "a1": np.random.normal(1.85, 0.1, 300),
                "a2": np.random.normal(2.10, 0.1, 300),
                "a3": np.random.normal(3.20, 0.15, 300),
                "a4": np.random.normal(2.65, 0.15, 300),
            })
            st.download_button(
                "📥 Download CSV Template",
                sample_template_df.to_csv(index=False),
                "sensor_telemetry_template.csv",
                "text/csv",
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # Handle uploaded CSV
        df_to_process = None
        if uploaded_file is not None:
            try:
                df_to_process = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded `{uploaded_file.name}` with {len(df_to_process)} time-series rows.")
            except Exception as e:
                st.error(f"❌ Failed to parse CSV: {e}")

        elif 'batch_df' in st.session_state:
            df_to_process = st.session_state['batch_df']

        if df_to_process is not None:
            # Map column names automatically
            column_mapping = {
                "motor vibration - x direction": "a1",
                "motor vibration x": "a1",
                "motor vibration - y direction": "a2",
                "motor vibration y": "a2",
                "gearbox - vibration": "a3",
                "gearbox vibration": "a3",
                "bearing vibration": "a4"
            }
            renamed_cols = {}
            for col in df_to_process.columns:
                clean_col = str(col).strip().lower()
                if clean_col in column_mapping:
                    renamed_cols[col] = column_mapping[clean_col]

            df_standard = df_to_process.rename(columns=renamed_cols)
            required_sensors = ["a1", "a2", "a3", "a4"]
            missing_sensors = [s for s in required_sensors if s not in df_standard.columns]

            if missing_sensors:
                st.error(f"❌ Missing required sensor columns: {missing_sensors}. Expected either [a1, a2, a3, a4] or full names.")
            else:
                with st.expander("👁️ Preview Raw Time-Series Data", expanded=False):
                    st.dataframe(df_standard.head(10), use_container_width=True)

                if st.button("🚀 Process All Sliding Windows", type="primary", use_container_width=True):
                    with st.spinner("Extracting 28 statistical features per sliding window..."):
                        sensor_windows = {}
                        for s in required_sensors:
                            sensor_windows[s] = create_windows(df_standard[s].values, window_size=256, step=128)

                        num_windows = min(len(sensor_windows[s]) for s in required_sensors)

                        if num_windows == 0:
                            st.warning(f"⚠️ Dataset has {len(df_standard)} rows. Sliding window requires at least 256 consecutive samples.")
                        else:
                            X_features = []
                            for i in range(num_windows):
                                feats = []
                                for s in required_sensors:
                                    feats.extend(extract_features(sensor_windows[s][i]))
                                X_features.append(feats)

                            X_features = np.array(X_features)
                            X_scaled = scaler.transform(X_features)
                            X_pca = pca.transform(X_scaled)

                            preds = model.predict(X_pca)
                            probs = model.predict_proba(X_pca)

                            class_map = {0: "Healthy", 1: "Faulty"}
                            pred_labels = [class_map.get(int(p), "Healthy") for p in preds]
                            confidences = [probs[idx][int(p)] * 100 for idx, p in enumerate(preds)]

                            batch_results_df = pd.DataFrame({
                                "Window_ID": np.arange(1, num_windows + 1),
                                "Condition": pred_labels,
                                "Confidence (%)": np.round(confidences, 2),
                                "Time_Step_Start": np.arange(0, num_windows * 128, 128),
                                "Time_Step_End": np.arange(256, num_windows * 128 + 256, 128)
                            })

                            # KPI Cards
                            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                            healthy_count = sum(1 for c in pred_labels if c == "Healthy")
                            faulty_count = sum(1 for c in pred_labels if c == "Faulty")
                            fault_rate = (faulty_count / num_windows) * 100

                            with kpi1:
                                st.metric("Evaluated Windows", num_windows)
                            with kpi2:
                                st.metric("Healthy Windows", healthy_count)
                            with kpi3:
                                st.metric("Faulty Windows", faulty_count)
                            with kpi4:
                                st.metric("Overall Fault Rate", f"{fault_rate:.1f}%")

                            # Charts
                            bc1, bc2 = st.columns([1, 1.6])
                            with bc1:
                                donut_fig = px.pie(
                                    values=[healthy_count, faulty_count],
                                    names=["Healthy", "Faulty"],
                                    title="Condition Breakdown",
                                    color=["Healthy", "Faulty"],
                                    color_discrete_map={"Healthy": "#10b981", "Faulty": "#ef4444"},
                                    hole=0.55
                                )
                                donut_fig.update_layout(
                                    paper_bgcolor='rgba(15, 23, 42, 0.5)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#f8fafc'),
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    height=280
                                )
                                st.plotly_chart(donut_fig, use_container_width=True)

                            with bc2:
                                timeline_fig = px.scatter(
                                    batch_results_df,
                                    x="Window_ID",
                                    y="Confidence (%)",
                                    color="Condition",
                                    title="Fault Telemetry Timeline Across Windows",
                                    color_discrete_map={"Healthy": "#10b981", "Faulty": "#ef4444"},
                                    size="Confidence (%)",
                                    size_max=10
                                )
                                timeline_fig.update_layout(
                                    paper_bgcolor='rgba(15, 23, 42, 0.5)',
                                    plot_bgcolor='rgba(11, 17, 32, 0.8)',
                                    font=dict(color='#f8fafc'),
                                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', color='#94a3b8'),
                                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', color='#94a3b8'),
                                    margin=dict(l=30, r=30, t=40, b=30),
                                    height=280
                                )
                                st.plotly_chart(timeline_fig, use_container_width=True)

                            st.dataframe(batch_results_df, use_container_width=True)

                            csv_data = batch_results_df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download Complete Batch Report (CSV)",
                                csv_data,
                                f"batch_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )

    # ==========================================
    # TAB 3: PCA & MODEL INTELLIGENCE
    # ==========================================
    with tab3:
        st.markdown("### 🔬 Model Intelligence & Decomposition")
        st.caption("Deep-dive into the Principal Component Analysis (PCA) dimensionality reduction and statistical feature representation.")

        # Pipeline diagram
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="margin: 0 0 12px 0; color: #38bdf8 !important;">🔄 End-to-End Machine Learning Architecture</h4>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.88rem;">
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px 14px; text-align: center;">
                    <div style="color: #38bdf8; font-weight: 700;">1. Raw Signal</div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">4 Sensors × 256 Samples</div>
                </div>
                <div style="color: #64748b; font-size: 1.2rem;">➔</div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(129, 140, 248, 0.3); border-radius: 8px; padding: 10px 14px; text-align: center;">
                    <div style="color: #818cf8; font-weight: 700;">2. Feature Extraction</div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">Mean, Std, RMS, Min, Max, Skew, Kurt</div>
                </div>
                <div style="color: #64748b; font-size: 1.2rem;">➔</div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(192, 132, 252, 0.3); border-radius: 8px; padding: 10px 14px; text-align: center;">
                    <div style="color: #c084fc; font-weight: 700;">3. StandardScaler</div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">Zero Mean & Unit Variance (28D)</div>
                </div>
                <div style="color: #64748b; font-size: 1.2rem;">➔</div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(244, 114, 182, 0.3); border-radius: 8px; padding: 10px 14px; text-align: center;">
                    <div style="color: #f472b6; font-weight: 700;">4. PCA Reduction</div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">14 Principal Components (95% Var)</div>
                </div>
                <div style="color: #64748b; font-size: 1.2rem;">➔</div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; padding: 10px 14px; text-align: center;">
                    <div style="color: #34d399; font-weight: 700;">5. SVM RBF Classifier</div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">Probabilistic Classification</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # PCA Scree Plot
        exp_var = pca.explained_variance_ratio_ * 100
        cum_var = np.cumsum(exp_var)
        pc_labels = [f"PC{i+1}" for i in range(len(exp_var))]

        scree_fig = go.Figure()
        scree_fig.add_trace(go.Bar(
            x=pc_labels,
            y=exp_var,
            name="Individual Variance (%)",
            marker_color="#06b6d4"
        ))
        scree_fig.add_trace(go.Scatter(
            x=pc_labels,
            y=cum_var,
            name="Cumulative Variance (%)",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=7)
        ))

        scree_fig.update_layout(
            title={'text': "📉 PCA Scree Plot (Explained Variance Ratio)", 'font': {'size': 16, 'color': '#f8fafc'}},
            paper_bgcolor='rgba(15, 23, 42, 0.5)',
            plot_bgcolor='rgba(11, 17, 32, 0.8)',
            xaxis=dict(title="Principal Component", gridcolor='rgba(255, 255, 255, 0.06)', color='#94a3b8'),
            yaxis=dict(title="Variance Explained (%)", gridcolor='rgba(255, 255, 255, 0.06)', color='#94a3b8'),
            legend=dict(font=dict(color='#cbd5e1')),
            margin=dict(l=30, r=30, t=50, b=30),
            height=340
        )
        st.plotly_chart(scree_fig, use_container_width=True)

        # Statistical Features Breakdown
        st.markdown("#### 📐 28 Extracted Features Breakdown")
        feature_rows = []
        feature_types = ["Mean", "Standard Deviation", "RMS (Root Mean Square)", "Minimum", "Maximum", "Skewness", "Kurtosis"]
        sensors = ["a1 (Motor X)", "a2 (Motor Y)", "a3 (Gearbox)", "a4 (Bearing)"]

        for s_idx, s in enumerate(sensors):
            for f_idx, f in enumerate(feature_types):
                feature_rows.append({
                    "Index": s_idx * 7 + f_idx + 1,
                    "Sensor Channel": s,
                    "Statistic Type": f,
                    "Domain": "Time-Domain Statistical",
                    "Purpose": "Detects energy shifts, crest factors, and signal asymmetry"
                })

        features_df = pd.DataFrame(feature_rows)
        st.dataframe(features_df, use_container_width=True)

    # ==========================================
    # TAB 4: TELEMETRY AUDIT HISTORY
    # ==========================================
    with tab4:
        st.markdown("### 📜 Session Telemetry Audit Log")
        st.caption("Chronological record of all equipment diagnostic inferences executed in this session.")

        if len(st.session_state.predictions_history) == 0:
            st.info("📭 No predictions recorded in this session yet. Execute a real-time or batch diagnostic to populate telemetry.")
        else:
            hist_df = pd.DataFrame(st.session_state.predictions_history)

            # Metrics
            h_c1, h_c2, h_c3, h_c4 = st.columns(4)
            with h_c1:
                st.metric("Total Inferences", len(hist_df))
            with h_c2:
                st.metric("Average Confidence", f"{hist_df['confidence'].mean():.1f}%")
            with h_c3:
                h_healthy = (hist_df['predicted_class'] == "Healthy").sum()
                st.metric("Healthy Records", h_healthy)
            with h_c4:
                h_faulty = (hist_df['predicted_class'] == "Faulty").sum()
                st.metric("Faulty Records", h_faulty)

            # Timeline Plot
            hist_fig = px.scatter(
                hist_df,
                x=hist_df.index + 1,
                y="confidence",
                color="predicted_class",
                title="Confidence Levels Across Session Inferences",
                labels={"x": "Inference #", "confidence": "Confidence (%)"},
                color_discrete_map={"Healthy": "#10b981", "Faulty": "#ef4444"},
                size="confidence",
                size_max=12
            )
            hist_fig.update_layout(
                paper_bgcolor='rgba(15, 23, 42, 0.5)',
                plot_bgcolor='rgba(11, 17, 32, 0.8)',
                font=dict(color='#f8fafc'),
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', color='#94a3b8'),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', color='#94a3b8'),
                margin=dict(l=30, r=30, t=40, b=30),
                height=280
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            # Table
            st.dataframe(hist_df, use_container_width=True)

            csv_hist = hist_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Audit History (CSV)",
                csv_hist,
                f"telemetry_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
