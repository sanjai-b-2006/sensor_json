import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Water Quality Analyzer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- LOAD CUSTOM CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    local_css("style.css")
except:
    pass # CSS from Stlite might load differently

# --- EXTERNAL DATA URLS (RAW) ---
JSON_URL = "https://raw.githubusercontent.com/sanjai-b-2006/sensor_json/main/sensor_data.json"
MODEL_URL = "https://raw.githubusercontent.com/sanjai-b-2006/sensor_json/main/edge_water_quality_model.pkl"

# --- LOAD DATA AND MODEL ---
@st.cache_data
def load_external_data(url):
    try:
        return pd.read_json(url)
    except Exception as e:
        st.error(f"Error loading JSON data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(url):
    try:
        import joblib
        import urllib.request
        with urllib.request.urlopen(url) as response:
            return joblib.load(response)
    except Exception as e:
        st.sidebar.warning(f"Note: ML Model could not be loaded from URL. using threshold logic for now. ({e})")
        return None

# Load initial data
initial_df = load_external_data(JSON_URL)

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state.logs = initial_df if not initial_df.empty else pd.DataFrame()

# Load Model
model = load_model(MODEL_URL)

# --- PREDICTION LOGIC ---
def predict_quality(tds, ph, turbidity, hardness, conductivity):
    """
    Predicts water quality using the loaded ML model.
    """
    # 1. Domestic Suitability (Always based on basic thresholds for safety)
    bathing_safe = 5.0 <= ph <= 9.0 and tds <= 1500
    washing_safe = hardness <= 500
    
    # 2. ML Model Prediction for Drinking
    ml_safe = None
    if model:
        input_data = np.array([[tds, ph, turbidity, hardness, conductivity]])
        try:
            # Simple assumption: 1 = Potable, 0 = Not Potable
            ml_safe = bool(model.predict(input_data)[0])
        except:
            ml_safe = None
            
    return ml_safe, bathing_safe, washing_safe

# --- HEADER SECTION ---
st.markdown("""
<div class='header-container'>
    <h1 class='main-title'>💧 AI Water Quality Analyzer</h1>
    <p class='sub-title'>Check if your water is safe to drink or use domestically powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --- CONTROL CENTER (Replaces Sidebar) ---
st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top: 0; color: #01579B;'>📋 Sensor Data Selection & Quick Metrics</h3>", unsafe_allow_html=True)

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 2, 1])

with ctrl_col1:
    if st.session_state.logs.empty:
        st.warning("No data found in the provided JSON link.")
        # Fallback values if no data
        tds, ph, turbidity, hardness, conductivity = 350, 7.2, 3.8, 120, 450
    else:
        # Create a list of labels for the dropdown
        log_options = st.session_state.logs.apply(
            lambda x: f"{x['timestamp']} | TDS: {int(x['TDS'])}", axis=1
        ).tolist()
        
        selected_log = st.selectbox(
            "Select a reading to analyze:", 
            log_options,
            index=0 # Default to the latest
        )
        
        # Extract selected record values
        idx = log_options.index(selected_log)
        record = st.session_state.logs.iloc[idx]
        
        tds = int(record["TDS"])
        ph = float(record["pH"])
        turbidity = float(record["Turbidity"])
        hardness = int(record["Hardness"])
        conductivity = int(record["Conductivity"])
        
        st.success(f"✅ Analyzing reading from: {record['timestamp']}")

with ctrl_col2:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1: st.metric("TDS", f"{tds} ppm")
    with m_col2: st.metric("pH", f"{ph}")
    with m_col3: st.metric("Turbidity", f"{turbidity}")
    with m_col4: st.metric("Hardness", f"{hardness}")
    with m_col5: st.metric("Conduc.", f"{conductivity}")
    st.markdown("</div>", unsafe_allow_html=True)

with ctrl_col3:
    st.write("") # Spacer
    st.write("") # Spacer
    if st.button("Refresh Data from GitHub", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True) # End Control Panel

# --- MAIN CONTENT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='title-bg bg-molecular'>🔘 Molecular Balance</div>", unsafe_allow_html=True)
    
    # Gauge for pH
    fig_ph = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = ph,
        title = {'text': "pH Level"},
        gauge = {
            'axis': {'range': [0, 14]},
            'steps': [
                {'range': [0, 6.5], 'color': "#ffebee"},
                {'range': [6.5, 8.5], 'color': "#e8f5e9"},
                {'range': [8.5, 14], 'color': "#ffebee"}
            ],
            'bar': {'color': "#00BCD4"}
        }
    ))
    fig_ph.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_ph, use_container_width=True)

    # Batch Summary Analysis
    if not st.session_state.logs.empty:
        df_eval = st.session_state.logs.copy()
        
        def calculate_score(r):
            res = predict_quality(r['TDS'], r['pH'], r['Turbidity'], r['Hardness'], r['Conductivity'])
            return res[0] if res[0] is not None else False

        safe_results = df_eval.apply(calculate_score, axis=1)
        health_score = (sum(safe_results) / len(df_eval)) * 100
        
        # Determine color based on score
        if health_score > 70:
            color_grad = "linear-gradient(135deg, #00C853 0%, #00E676 100%)" # Green
            text_color = "white"
        elif health_score >= 40:
            color_grad = "linear-gradient(135deg, #FFD600 0%, #FFEA00 100%)" # Yellow
            text_color = "#333"
        else:
            color_grad = "linear-gradient(135deg, #D50000 0%, #FF1744 100%)" # Red
            text_color = "white"

        st.markdown(f"""
        <div class='stCard' style='background: {color_grad} !important; color: {text_color} !important; text-align: center; border: none; box-shadow: 0 10px 40px rgba(0,0,0,0.15) !important;'>
            <h4 style='color: {text_color}; margin:0; opacity: 0.9; font-size: 0.9rem;'>🌐 DATASET HEALTH SCORE</h4>
            <div style='font-size: 3.5rem; font-weight: 900; margin: 10px 0; letter-spacing: -2px;'>{health_score:.1f}%</div>
            <p style='font-size: 0.85rem; margin:0; font-weight: 600;'>Analyzed {len(df_eval)} JSON records</p>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart for other parameters
    st.write("")
    st.markdown("<div class='title-bg bg-distribution'>📊 Metric Distribution</div>", unsafe_allow_html=True)
    params_df = pd.DataFrame({
        'Parameter': ['TDS', 'Hardness', 'Conductivity'],
        'Value': [tds, hardness, conductivity],
        'Max WHO': [600, 200, 1500]
    })
    fig_bar = px.bar(params_df, x='Parameter', y='Value', color='Value', 
                     color_continuous_scale='Tealgrn')
    fig_bar.update_layout(height=300, title_text="")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("<div class='title-bg bg-analysis'>🤖 Neural Analysis</div>", unsafe_allow_html=True)
    
    ml_safe, b_safe, w_safe = predict_quality(tds, ph, turbidity, hardness, conductivity)
    
    # Drinking Result Card
    st.markdown(f"""
    <div class='stCard'>
        <div style='display: flex; justify-content: space-between;'>
            <h4 style='margin:0;'>AI Potability Prediction</h4>
            <span class='status-badge {"safe" if ml_safe else "unsafe"}'>
                {"SAFE" if ml_safe else "UNSAFE"}
            </span>
        </div>
        <div style='margin-top: 15px; font-size: 1rem; font-weight: 600;'>
            {"✅ The water is predicted to be potable." if ml_safe else "❌ The water is predicted to be non-potable." if ml_safe is False else "⚠️ Prediction model is currently unavailable."}
        </div>
        <p style='margin-top: 10px; font-size: 0.8rem; color: #666;'>
            This result is generated by the Edge Water Quality ML Model using real-time sensor inputs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    # Domestic Result Cards
    st.markdown("<div class='title-bg bg-domestic'>📎 Domestic Use Suitability</div>", unsafe_allow_html=True)
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        st.markdown(f"""
        <div class='stCard' style='text-align: center; padding: 15px !important;'>
            <img src="https://img.icons8.com/ios/50/00BCD4/shower.png" width="30"/>
            <p style='margin: 10px 0;'>Bathing</p>
            <span class='status-badge {"safe" if b_safe else "caution"}'>
                {"Safe" if b_safe else "Caution"}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    with subcol2:
        st.markdown(f"""
        <div class='stCard' style='text-align: center; padding: 15px !important;'>
            <img src="https://img.icons8.com/ios/50/00BCD4/washing-machine.png" width="30"/>
            <p style='margin: 10px 0;'>Washing</p>
            <span class='status-badge {"safe" if w_safe else "caution"}'>
                {"Safe" if w_safe else "Caution"}
            </span>
        </div>
        """, unsafe_allow_html=True)

# --- LOGS SECTION ---
st.divider()
st.markdown("<div class='title-bg bg-history'>📜 Historical Intelligence Log</div>", unsafe_allow_html=True)

# Format and display logs
display_df = st.session_state.logs.copy()
# Add result columns based on ML logic
def get_status(row):
    ml, _, _ = predict_quality(row['TDS'], row['pH'], row['Turbidity'], row['Hardness'], row['Conductivity'])
    return "✅ Safe" if ml else "❌ Unsafe"

display_df['AI Prediction'] = display_df.apply(get_status, axis=1)

st.dataframe(display_df, use_container_width=True)

# AI Badge footer
st.markdown("""
<div style='text-align: center; margin-top: 30px;'>
    <span style='background: #E0F7FA; color: #01579B; padding: 5px 15px; border-radius: 50px; font-size: 0.8rem; font-weight: bold;'>
        ⚡ POWERED BY ANALYTICS CLOUD
    </span>
</div>
""", unsafe_allow_html=True)
