import matplotlib
matplotlib.use('Agg') # Forces the Mac to save the file without needing a window
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Load Data
try:
    df = pd.read_csv('sensor_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print("Step 1: Data loaded from CSV.")
except Exception as e:
    print(f"Error: Could not load CSV. {e}")
    exit()

# 2. Create Plot
print("Step 2: Generating the dashboard...")
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Temperature Plot
ax1.plot(df['Timestamp'], df['Temperature_C'], color='#ff6600', label='Temp (°C)')
ax1.axhline(y=85, color='red', linestyle='--')
failures = df[df['Failure_Risk'] == 1]
ax1.scatter(failures['Timestamp'], failures['Temperature_C'], color='red', marker='x', label='Failure')
ax1.set_title('5V Circuit Predictive Maintenance')
ax1.legend()

# Vibration Plot
ax2.plot(df['Timestamp'], df['Vibration_mm_s'], color='#00ccff', label='Vibration (mm/s)')
ax2.axhline(y=2.0, color='yellow', linestyle='--')
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()

# 3. Save and Verify
file_name = 'failure_analysis.png'
plt.savefig(file_name, dpi=300)
print(f"Step 3: Attempted to save {file_name}")

if os.path.exists(file_name):
    print(f"✅ FINAL SUCCESS: {file_name} is now in your folder at {os.getcwd()}")
else:
    print("❌ ERROR: File was not created.")