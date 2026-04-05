import pandas as pd
import numpy as np
import datetime

def generate_sensor_data(rows=100):
    np.random.seed(42)
    
    data = {
        'Timestamp': [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(rows)],
        'Voltage_V': np.random.uniform(3.5, 5.2, rows),
        'Temperature_C': np.random.uniform(30, 95, rows),
        'Vibration_mm_s': np.random.uniform(0.1, 2.5, rows)
    }
    
    df = pd.DataFrame(data)
    
    # AI Labeling Logic
    df['Failure_Risk'] = np.where((df['Temperature_C'] > 85) & (df['Vibration_mm_s'] > 2.0), 1, 0)
    
    return df

if __name__ == "__main__":
    raw_data = generate_sensor_data(200)
    print(raw_data.head())
    raw_data.to_csv('sensor_data.csv', index=False)
