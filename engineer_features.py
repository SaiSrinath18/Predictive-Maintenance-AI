import pandas as pd
import numpy as np

# Load Day 1 Data
df = pd.read_csv('sensor_data.csv')

# 1. Calculate Gradient (Rate of Temperature Change)
df['Temp_Gradient'] = df['Temperature_C'].diff().fillna(0)

# 2. Calculate Moving Average (Smoothing out 5V sensor noise)
df['Temp_MA_5'] = df['Temperature_C'].rolling(window=5).mean()

# 3. Predictive Early Warning Logic
# If (Current Temp + 2 seconds of current Gradient) > 85, Trigger Warning
df['Early_Warning'] = ((df['Temperature_C'] + (df['Temp_Gradient'] * 2)) >= 85).astype(int)

# Save the "Smart" Dataset
df.dropna().to_csv('processed_data.csv', index=False)
print("✅ Step 1 Complete: 'processed_data.csv' created with Predictive Features.")
