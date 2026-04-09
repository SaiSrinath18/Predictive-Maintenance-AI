import joblib
import pandas as pd
import numpy as np
import time
import os

# 1. Load the "Brain" (Trained Model)
# Ensure maintenance_model.pkl is in the same folder!
try:
    model = joblib.load('maintenance_model.pkl')
    print("✅ AI Brain Loaded Successfully.")
except FileNotFoundError:
    print("❌ Error: maintenance_model.pkl not found. Run Day 4 first!")
    exit()

def simulate_live_sensor():
    """
    Simulates a single data point from a 5V circuit.
    Generates variations in Temp and Gradient to test the AI.
    """
    # Randomly simulate stable vs. overheating scenarios
    if np.random.rand() > 0.8:
        temp = np.random.uniform(80, 95)  # Potential Danger
        grad = np.random.uniform(0.6, 1.5) 
    else:
        temp = np.random.uniform(50, 75)  # Stable Operating
        grad = np.random.uniform(0.1, 0.4)
        
    ma = temp * 0.99 # Simplified Moving Average
    return pd.DataFrame([[temp, grad, ma]], columns=['Temperature_C', 'Temp_Gradient', 'Temp_MA_5'])

print("🚀 Starting Real-Time Predictive Maintenance Engine...")
print("📝 Logging alerts to 'maintenance_logs.txt'...")
print("-" * 60)

try:
    while True:
        # Get 'new' data reading
        current_data = simulate_live_sensor()
        
        # AI Logic: Predict State (0 or 1) and Probability (0.0 to 1.0)
        prediction = model.predict(current_data)[0]
        probability = model.predict_proba(current_data)[0][1] 
        
        timestamp = time.strftime("%H:%M:%S")
        temp_val = current_data['Temperature_C'].values[0]
        
        # UI Formatting
        if prediction == 1:
            status = "🚨 DANGER"
            # --- THE LOGGING SYSTEM ---
            with open("maintenance_logs.txt", "a") as log:
                log_entry = f"ALERT | {time.ctime()} | Temp: {temp_val:.2f}C | Risk: {probability:.2%}\n"
                log.write(log_entry)
        else:
            status = "✅ STABLE"

        # Print to Terminal for Live Monitoring
        print(f"[{timestamp}] Temp: {temp_val:.2f}°C | Risk Score: {probability:>7.2%} | Status: {status}")
        
        time.sleep(1) # Monitor every 1 second

except KeyboardInterrupt:
    print("\n🛑 Monitoring Stopped by User.")
    print(f"📂 Check 'maintenance_logs.txt' for the incident history.")
