import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")
st.title("📡 Industrial AI: Real-Time Signal Monitoring & Diagnostics")
st.markdown("---")

# 2. Load the Day 5 'Brain'
@st.cache_resource
def load_model():
    return joblib.load('maintenance_model.pkl')

model = load_model()

# 3. Sidebar for System Info
st.sidebar.header("System Status")
st.sidebar.info("Model: Random Forest Classifier\n\nInput Source: 5V Rail Sensor Simulation")
st.sidebar.markdown("Created by: **SaiSrinath18**")

# 4. Create placeholders for the dynamic UI
kpi_placeholder = st.empty()
chart_placeholder = st.empty()

# 5. Initialize Data Buffer for the Chart
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=['Time', 'Temp', 'Risk'])

# 6. The Real-Time Logic Loop
while True:
    # --- SIMULATION ENGINE ---
    # We need 3 features to match your trained model's requirements
    temp = np.random.uniform(30, 95)
    grad = np.random.uniform(-5, 5)
    moving_avg = temp + np.random.uniform(-2, 2) # The 3rd missing feature
    
    # --- INFERENCE ENGINE ---
    # Passing all 3 features in the exact order the model expects
    risk_prob = model.predict_proba([[temp, grad, moving_avg]])[0][1] * 100
    
    # Update History
    new_data = pd.DataFrame({
        'Time': [datetime.now().strftime("%H:%M:%S")],
        'Temp': [temp],
        'Risk': [risk_prob]
    })
    
    # Keep only the last 20 data points for a smooth chart
    st.session_state.data_history = pd.concat([st.session_state.data_history, new_data]).tail(20)

    # --- UI RENDERING ---
    with kpi_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Real-Time Temperature", f"{temp:.2f} °C")
        col2.metric("Inference Risk Score", f"{risk_prob:.1f}%")
        
        # System Health Indicator
        if risk_prob > 80:
            col3.error("🚨 CRITICAL: FAILURE IMMINENT")
        elif risk_prob > 50:
            col3.warning("⚠️ WARNING: ANOMALY")
        else:
            col3.success("✅ SYSTEM NOMINAL")

    with chart_placeholder.container():
        st.subheader("Live Diagnostic Telemetry")
        # Plotting Temperature and Risk Score over time
        st.line_chart(st.session_state.data_history.set_index('Time'))

    time.sleep(1) # Refresh rate: 1Hz
