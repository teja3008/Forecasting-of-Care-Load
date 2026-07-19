# Forecasting-of-Care-Load
analysis of care load in hhs centers
# 📊 Predictive Forecasting of Care Load & Placement Demand

### AI-Powered Forecasting Dashboard for HHS Child Welfare Capacity Planning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange.svg)]()
[![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-green.svg)]()

## 📌 Overview

Predictive Forecasting of Care Load & Placement Demand is a Machine Learning and Time Series forecasting application developed to assist stakeholders in anticipating the number of children under HHS care based on historical operational data.

Instead of relying solely on historical reports, this project provides actionable forecasts that support proactive staffing, shelter capacity planning, and operational decision-making through an interactive Streamlit dashboard. The application forecasts HHS care load using historical intake, transfer, and discharge data. :contentReference[oaicite:0]{index=0}

---

## 🚀 Live Demo

**Streamlit Application**

https://forecasting-of-care-load-zdh72ndvu9hs5jsuxq9ngv.streamlit.app/

---

# ✨ Features

- 📈 Interactive time-series visualization
- 🤖 Multiple forecasting models
  - Random Forest Regressor
  - ARIMA
  - Naive Forecast
- 📊 Model performance comparison
- 📉 Forecast horizon selection (7, 14 and 30 days)
- 📌 Capacity breach probability estimation
- 📍 Forecast stability analysis
- 📅 Walk-forward validation
- 📊 Historical trend analysis
- ⚠ Early warning system using Net System Pressure
- 📈 KPI Dashboard for decision support

The dashboard also highlights structural shifts in the data and compares forecast performance across multiple horizons. :contentReference[oaicite:1]{index=1}

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Statsmodels
- Random Forest Regression
- ARIMA Time Series Model
- Matplotlib
- Machine Learning
- Time Series Forecasting

---

# 📂 Project Structure

```
Forecasting-of-Care-Load/
│
├── app.py
├── hhs_sheet.csv
├── requirements.txt
├── Research Paper.pdf
├── Executive Summary.pdf
├── README.md
└── assets/
```

---

# 📊 Dataset

The project uses daily operational reporting data from the **HHS Unaccompanied Alien Children (UAC) Program** covering approximately **three years (January 2023 – December 2025)**.

The dataset includes:

- Children apprehended in CBP custody
- Children currently in CBP custody
- Children transferred to HHS care
- Children currently in HHS care
- Children discharged from HHS care

The data required cleaning due to mixed date formats, numeric formatting, and trailing blank rows before feature engineering. :contentReference[oaicite:2]{index=2}

---

# ⚙ Feature Engineering

The forecasting model uses several engineered features including:

- Lag Features
  - lag1
  - lag7
  - lag14

- Rolling Statistics
  - 7-day rolling mean
  - 14-day rolling mean
  - Rolling standard deviation

- Net System Pressure

- Report Gap

- Day of Week

These features improve forecasting performance by capturing trends, volatility, reporting cadence, and operational pressure. :contentReference[oaicite:3]{index=3}

---

# 🤖 Machine Learning Models

The following forecasting models were evaluated:

| Model | Purpose |
|--------|----------|
| Random Forest | Final production model |
| ARIMA | Statistical forecasting |
| Naive Forecast | Baseline comparison |

Random Forest consistently outperformed statistical baselines in walk-forward validation. :contentReference[oaicite:4]{index=4}

---

# 📈 Model Performance

### Random Forest Results

| Metric | Value |
|---------|-------|
| Forecast Accuracy | **91.6%** |
| MAE | **211.22** |
| RMSE | **361.59** |
| MAPE | **8.40%** |
| Capacity Breach Probability | **0%** |

The study found that Random Forest reduced forecasting error by approximately **28–32%** compared with statistical and naive baseline models. :contentReference[oaicite:5]{index=5}

---

# 📊 Dashboard Includes

✔ Historical Care Load Trends

✔ Forecast Comparison

✔ Forecast Horizon Selection

✔ KPI Cards

✔ Capacity Risk Analysis

✔ Model Comparison

✔ Forecast Error Analysis

✔ Net System Pressure Visualization

✔ Walk-forward Validation

---

# 🎯 Key Findings

- Random Forest achieved the highest forecasting accuracy.
- One-day forecasts are significantly more reliable than longer forecast horizons.
- A major structural change occurred around mid-2024, affecting care load patterns.
- Recent forecasts indicate low probability of capacity breaches.
- Net System Pressure is an effective early-warning operational indicator. :contentReference[oaicite:6]{index=6}

---

# ▶ Installation

Clone the repository

```bash
git clone https://github.com/teja3008/Forecasting-of-Care-Load.git
```

Navigate to project directory

```bash
cd Forecasting-of-Care-Load
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 💡 Future Enhancements

- LSTM-based forecasting
- XGBoost Regression
- Prophet Forecasting
- Real-time API integration
- Automated model retraining
- Explainable AI (SHAP)
- Cloud deployment with CI/CD

Some of these enhancements, including additional model comparisons and deployment, are also identified as future work in the research paper. :contentReference[oaicite:7]{index=7}

---

# 👨‍💻 Author

**Teja**

Computer Science Engineering Student

Machine Learning | Data Analytics | Time Series Forecasting | Streamlit

GitHub: https://github.com/teja3008


---

# 📜 License

This project is intended for educational and research purposes.

---

⭐ If you found this project useful, don't forget to Star the repository!
