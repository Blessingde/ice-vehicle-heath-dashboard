# ICE Vehicle Health Prediction Dashboard

## 📌 Project Overview

This project presents an **AI-powered Vehicle Health Prediction Dashboard** designed to transform reactive maintenance into proactive, predictive maintenance for Internal Combustion Engine (ICE) vehicles. By leveraging machine learning and historical sensor data, the system monitors the real-time health of critical components, predicts their Remaining Useful Life (RUL), Time to Failure (TTF), and Failure Probability, and provides actionable maintenance recommendations.

The primary objective is to enhance vehicle safety and performance while significantly minimizing unexpected breakdowns and operational downtime.

## ✨ Features

- **Comprehensive Component Monitoring:** Get health insights for three critical vehicle systems:
  - **Engine System️**
  - **Brake System**
  - **Tire System**
- **Intuitive Data Upload:** Easily upload your vehicle's time-series sensor data via CSV files.
- **Advanced Predictive Metrics:** For each component, the dashboard displays:
  - **Health Score (0-100%):** A continuous indicator of the component's current condition.
  - **Failure Risk (Next 30 Days):** The probability of a component failure within the upcoming month.
  - **Remaining Useful Life (RUL in days):** An estimate of the days until the component requires maintenance.
  - **Time To Failure (TTF in days):** An estimate of the days until a critical failure event.
- **Intelligent Maintenance Advice:** Receive specific, actionable recommendations based on the predicted health metrics and predefined thresholds, enabling efficient maintenance planning.

## ⚙️ How It Works

1.  **Data Ingestion:** Upload a CSV file containing your vehicle's sensor readings and operational data.
2.  **Vehicle Selection:** Choose a specific `Vehicle_ID` from the dropdown list for analysis.
3.  **Feature Processing:** The system processes the raw data, calculating various time-series features (like rolling averages and lagged values) that represent the recent operational history of the vehicle.
4.  **Machine Learning Prediction:** Pre-trained multi-output `RandomForestRegressor` models analyze the processed features to predict the RUL, TTF, Failure Probability, and Health Score for each monitored component.
5.  **Dashboard Visualization:** The dashboard displays these predictions and offers clear, actionable maintenance advice.

### 🛠️ Technologies Used

- **Python**
- **Streamlit:** For building the interactive web dashboard.
- **Pandas:** For efficient data manipulation and analysis.
- **Scikit-learn:** For machine learning models (`RandomForestRegressor`) and data preprocessing.
- **Joblib:** For efficient serialization and deserialization of trained models and preprocessors.

## 🚀 Usage

### 1. Clone the Repository

### 1. Clone the Repository

```bash
git clone https://github.com/Blessingde/ice-vehicle-heath-dashboard.git
cd -ice-vehicle-heath-dashboard-
```

### 2. Run the Streamlit app:

```bash

streamlit run app.py

```

### 📦Installation

```bash

pip install requirements.txt

```

## Demo link

[Streamlit App](https://ice-vehicle-heath-dashboard.streamlit.app/)

### 📊 Sample Dataset

Use this [sample CSV file](https://drive.google.com/file/d/1Gb9XDd6ZtCbDlhQ1k1H5ZQHDXvI4vVaO/view?usp=sharing) to test the engine prediction system.
