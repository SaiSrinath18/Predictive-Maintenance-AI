import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Load the Brain and the Data
model = joblib.load('maintenance_model.pkl')
df = pd.read_csv('processed_data.csv')

# 2. Setup the Visualization (Temp vs Gradient)
plt.figure(figsize=(10, 6))

# 3. Plot the actual data points
# Points where the AI predicts "Safe"
safe = df[df['Early_Warning'] == 0]
danger = df[df['Early_Warning'] == 1]

plt.scatter(safe['Temperature_C'], safe['Temp_Gradient'], c='blue', alpha=0.3, label='Actual Safe Data')
plt.scatter(danger['Temperature_C'], danger['Temp_Gradient'], c='orange', alpha=0.5, label='Actual Danger Data')

# 4. Add the AI's "Logic Lines"
plt.axvline(x=85, color='red', linestyle='--', label='Critical Temp (85°C)')
plt.axhline(y=0.5, color='purple', linestyle='--', label='High Heat Velocity')

plt.title("Day 4: AI Decision Mapping (Random Forest Logic)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Heat Velocity (Gradient)")
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Save the physical image
plt.savefig('day4_ai_logic.png')
print("✅ Physical output saved as 'day4_ai_logic.png'!")
plt.show()
