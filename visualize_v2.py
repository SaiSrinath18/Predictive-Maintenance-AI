import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd

# Load the "Smart" Data
df = pd.read_csv('processed_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top Plot: Temperature & Early Warnings
ax1.plot(df['Timestamp'], df['Temperature_C'], color='#ff6600', alpha=0.4, label='Raw Temp')
ax1.plot(df['Timestamp'], df['Temp_MA_5'], color='#ffcc00', label='Smoothed (MA)')
ax1.axhline(y=85, color='red', linestyle='--', label='Threshold (85°C)')

# Shading the Early Warning Areas
warnings = df[df['Early_Warning'] == 1]
for i in warnings.index:
    ax1.axvspan(df['Timestamp'].iloc[i], df['Timestamp'].iloc[i], color='red', alpha=0.2)

ax1.set_title('Predictive Maintenance: Early Warning Detection')
ax1.legend()

# Bottom Plot: Thermal Velocity (Gradient)
ax2.plot(df['Timestamp'], df['Temp_Gradient'], color='#00ff99', label='Temp Gradient (ΔT)')
ax2.set_title('Thermal Velocity (Rate of Change)')
ax2.legend()

plt.tight_layout()
plt.savefig('predictive_analysis.png')
print("✅ Step 2 Complete: 'predictive_analysis.png' saved.")
