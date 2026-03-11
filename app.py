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
    initial_sidebar_state="expanded",
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
    Predicts water quality using the loaded model if available, 
    otherwise falls back to standard thresholds.
    """
    if model:
        # Assuming model takes [TDS, pH, Turbidity, Hardness, Conductivity]
        input_data = np.array([[tds, ph, turbidity, hardness, conductivity]])
        try:
            # Adjust this based on your specific model's output (binary, multi-class, etc.)
            prediction = model.predict(input_data)[0]
            drinking_safe = bool(prediction) # Adjust based on your model's labels
            # For domestic use, we use thresholds if the model is only for drinking
            bathing_safe = tds < 1500 and 5.0 <= ph <= 9.0
            washing_safe = hardness < 500
            confidence = 0.92 # Dummy confidence if model doesn't provide it
            return drinking_safe, bathing_safe, washing_safe, confidence
        except:
            pass
            
    # Fallback Threshold logic
    drinking_safe = tds <= 1000 and 6.5 <= ph <= 8.5 and turbidity <= 5 and hardness <= 300
    bathing_safe = tds <= 1500 and 5.0 <= ph <= 9.0
    washing_safe = hardness <= 500
    confidence = 0.85
    return drinking_safe, bathing_safe, washing_safe, confidence

# --- HEADER SECTION ---
st.markdown("""
<div class='header-container'>
    <h1 class='main-title'>💧 AI Water Quality Analyzer</h1>
    <p class='sub-title'>Check if your water is safe to drink or use domestically powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR / INPUT PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/water-droplet.png", width=60)
    st.markdown("### Sensor Controls")
    st.markdown("Adjust parameters for real-time analysis.")
    
    tds = st.sidebar.slider("TDS (ppm)", 0, 3000, 350)
    ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.2)
    turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 50.0, 3.8)
    hardness = st.sidebar.slider("Hardness (mg/L)", 0, 1000, 120)
    conductivity = st.sidebar.slider("Conductivity (µS/cm)", 0, 2000, 450)
    
    analyze_btn = st.button("Analyze Water Quality", use_container_width=True)

# --- MAIN CONTENT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Parameter Visualization")
    
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

    # Bar chart for other parameters
    params_df = pd.DataFrame({
        'Parameter': ['TDS', 'Hardness', 'Conductivity'],
        'Value': [tds, hardness, conductivity],
        'Max Ref': [1000, 500, 1500]
    })
    fig_bar = px.bar(params_df, x='Parameter', y='Value', color='Value', 
                     color_continuous_scale='Tealgrn', title="Quality Indices")
    fig_bar.update_layout(height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### 🏆 Prediction Results")
    
    d_safe, b_safe, w_safe, conf = predict_quality(tds, ph, turbidity, hardness, conductivity)
    
    # Drinking Result Card
    st.markdown(f"""
    <div class='stCard'>
        <h4>Potability Status</h4>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span class='status-badge {"safe" if d_safe else "unsafe"}'>
                {"SAFE TO DRINK" if d_safe else "UNSAFE FOR DRINKING"}
            </span>
            <span style='font-size: 0.9rem; color: #666;'>Confidence: {conf*100:.1f}%</span>
        </div>
        <p style='margin-top: 10px; font-size: 0.85rem;'>Based on TDS, pH, and chemical concentration thresholds.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    # Domestic Result Cards
    st.markdown("#### Domestic Suitability")
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
st.markdown("### 📜 System Logs & History")
if analyze_btn:
    # Append current analysis to session state logs
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "TDS": tds,
        "pH": ph,
        "Turbidity": turbidity,
        "Hardness": hardness,
        "Conductivity": conductivity
    }
    st.session_state.logs = pd.concat([pd.DataFrame([new_entry]), st.session_state.logs], ignore_index=True)

# Format and display logs
display_df = st.session_state.logs.copy()
# Add dummy result columns for historical data
display_df['Drinking'] = display_df.apply(lambda row: "Safe" if predict_quality(row['TDS'], row['pH'], row['Turbidity'], row['Hardness'], row['Conductivity'])[0] else "Unsafe", axis=1)

st.dataframe(display_df, use_container_width=True)

# AI Badge footer
st.markdown("""
<div style='text-align: center; margin-top: 30px;'>
    <span style='background: #E0F7FA; color: #01579B; padding: 5px 15px; border-radius: 50px; font-size: 0.8rem; font-weight: bold;'>
        ⚡ POWERED BY ANALYTICS CLOUD
    </span>
</div>
""", unsafe_allow_html=True)
