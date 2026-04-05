# Predictive Maintenance AI for Hardware Systems

## Overview
This project simulates an Industrial Internet of Things (IIoT) environment where sensor data (Voltage, Temperature, Vibration) is monitored in real-time. The goal is to use AI to predict hardware failure before it happens.

## Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas (Data Manipulation), NumPy (Numerical Logic)
- **Domain:** Electronics & Communication Engineering (ECE)

## How it Works
1. **Data Generation:** Simulates synthetic sensor streams for a 5V circuit.
2. **Failure Logic:** Implements a threshold-based risk assessment (Supervised Learning approach).
3. **Export:** Generates a `sensor_data.csv` for future Machine Learning model training.

## Day 1 Progress
- Set up local environment in mac mini m4.
- Established Git version control.
- Built the initial data pipeline.

**Phase 2: Exploratory Data Analysis (EDA) & Signal Visualization**

## 🛠 Day 2 Milestone: Visualization & Validation
Today’s focus was transitioning from raw data collection to **Diagnostic Visualization**. Using `Matplotlib`, I built a synchronized telemetry dashboard to validate the failure logic established in Day 1.

### Key Technical Achievements:
* **Time-Series Synchronization:** Implemented dual-axis subplots to align Temperature (°C) and Vibration (mm/s) data on a shared time-axis (`sharex=True`).
* **Threshold Mapping:** Integrated horizontal threshold overlays ($> 85°C$ and $> 2.0mm/s$) to visually identify "At-Risk" zones.
* **Anomaly Detection Marking:** Programmed automatic scatter-plot markers ('X') to highlight exactly where the system logic flags a potential hardware failure.

### 📊 Diagnostic Dashboard
![Failure Analysis](failure_analysis.png)
*The graph above confirms that our failure risk triggers align perfectly with thermal spikes.*

---

## 🧠 AIML Learning Journal
As a 2nd-year ECE student bridging into AI, today's core learnings included:
1. **Data Distribution:** Understanding how often the system enters a "Failure State" (Class Imbalance).
2. **Signal-to-Logic Mapping:** Verifying that the mathematical logic in `main.py` matches the physical behavior shown in the graphs.
3. **Efficiency in Data:** Practiced the **Two-Pointer Technique** for efficient array searching—essential for real-time sensor processing.

---

## 🚀 How to Run (Day 2)
1. Ensure `sensor_data.csv` exists in the root directory.
2. Run the visualization script:
   ```bash
   python3 visualize.py
3.Check the directory for failure_analysis.png to see the generated report.