import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

st.set_page_config(page_title="HHS Care Load Forecast", layout="wide")
st.title("Predictive Forecasting: HHS Care Load & Placement Demand")
st.write("Forecasting the number of children in HHS care based on historical intake, transfer, and discharge patterns.")

# ---------- Load and clean data ----------
@st.cache_data
def load_data():
    df = pd.read_csv('hhs_sheet.csv')
    df = df.dropna(subset=['Date']).copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df['Children in HHS Care'] = df['Children in HHS Care'].str.replace(',', '', regex=False).astype(float)
    df = df.sort_values('Date').reset_index(drop=True)

    df['report_gap'] = (df['Date'] - df['Date'].shift(1)).dt.days
    df['lag1'] = df['Children in HHS Care'].shift(1)
    df['lag7'] = df['Children in HHS Care'].shift(7)
    df['lag14'] = df['Children in HHS Care'].shift(14)
    df['rolling_mean_7'] = df['Children in HHS Care'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['Children in HHS Care'].shift(1).rolling(window=7).std()
    df['rolling_mean_14'] = df['Children in HHS Care'].shift(1).rolling(window=14).mean()
    df['rolling_std_14'] = df['Children in HHS Care'].shift(1).rolling(window=14).std()
    df['net_stays'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']
    df['day_of_week_num'] = df['Date'].dt.dayofweek

    return df

df = load_data()
target = 'Children in HHS Care'
features = ['lag1', 'lag7', 'lag14', 'rolling_mean_7', 'rolling_std_7',
            'rolling_mean_14', 'rolling_std_14', 'net_stays', 'day_of_week_num', 'report_gap']

df_model = df.dropna().reset_index(drop=True)

n_splits = 5
test_window = 30

# ---------- Overview chart ----------
st.subheader("Children in HHS Care over time")
st.line_chart(df.set_index('Date')[target])
st.info("A sustained structural drop is visible around mid-2024 — the series shifts from roughly "
        "8,000\u201310,000 down to roughly 2,000\u20134,000, while the weekly reporting rhythm continues "
        "at the new, lower scale. This shift affects historical model accuracy and is worth stakeholder investigation.")

st.subheader("Net system pressure (net_stays)")
st.caption("Children transferred out of CBP custody minus children discharged from HHS care. "
           "Positive values mean intake is outpacing discharge that day.")
st.line_chart(df.set_index('Date')['net_stays'])

st.divider()

# ---------- KPI summary (fixed, computed once with walk-forward validation) ----------
@st.cache_data
def compute_kpis():
    mae_scores, rmse_scores, mape_scores = [], [], []
    for i in range(n_splits):
        split_point = len(df_model) - test_window * (n_splits - i)
        train_fold = df_model.iloc[:split_point]
        test_fold = df_model.iloc[split_point: split_point + test_window]

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(train_fold[features], train_fold[target])
        y_pred = model.predict(test_fold[features])
        y_true = test_fold[target]

        mae_scores.append(mean_absolute_error(y_true, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape_scores.append(mean_absolute_percentage_error(y_true, y_pred))

    capacity_threshold = df_model[target].quantile(0.90)
    breach_probs = []
    for i in range(n_splits):
        split_point = len(df_model) - test_window * (n_splits - i)
        train_fold = df_model.iloc[:split_point]
        test_fold = df_model.iloc[split_point: split_point + test_window]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(train_fold[features], train_fold[target])
        y_pred = model.predict(test_fold[features])
        breach_probs.append((y_pred > capacity_threshold).mean())

    horizon_results = {}
    for h in [1, 7, 14]:
        train_fold = df_model.iloc[:-h]
        test_fold = df_model.iloc[-h:]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(train_fold[features], train_fold[target])
        y_pred = model.predict(test_fold[features])
        horizon_results[h] = mean_absolute_error(test_fold[target], y_pred)

    return {
        "mae": np.mean(mae_scores),
        "rmse": np.mean(rmse_scores),
        "mape": np.mean(mape_scores),
        "accuracy": 1 - np.mean(mape_scores),
        "stability": np.std(mae_scores),
        "breach_prob": np.mean(breach_probs),
        "capacity_threshold": capacity_threshold,
        "horizon": horizon_results,
    }

kpis = compute_kpis()

st.subheader("Model performance summary (Random Forest, walk-forward validated)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{kpis['mae']:.1f}")
c2.metric("RMSE", f"{kpis['rmse']:.1f}")
c3.metric("MAPE", f"{kpis['mape']:.2%}")
c4.metric("Forecast Accuracy", f"{kpis['accuracy']:.2%}")

c5, c6, c7 = st.columns(3)
c5.metric("Forecast Stability Index", f"{kpis['stability']:.1f}", help="Std. dev. of MAE across 5 walk-forward rounds. Lower = more consistent.")
c6.metric("Capacity Breach Probability", f"{kpis['breach_prob']:.2%}", help=f"Share of forecasts exceeding the 90th-percentile threshold ({kpis['capacity_threshold']:.0f} children)")
c7.metric("Capacity Threshold (90th pct.)", f"{kpis['capacity_threshold']:.0f}")

st.subheader("Forecast error by horizon")
horizon_df = pd.DataFrame.from_dict(kpis['horizon'], orient='index', columns=['MAE'])
horizon_df.index = [f"t+{h}" for h in horizon_df.index]
st.bar_chart(horizon_df)
st.caption("Forecasts are most reliable at 1 day ahead; accuracy degrades at 7- and 14-day horizons.")

st.divider()

# ---------- Sidebar controls ----------
st.sidebar.header("Forecast settings")

horizon = st.sidebar.selectbox("Forecast horizon", options=[7, 14, 30], index=0)
model_choice = st.sidebar.selectbox("Model", options=["Random Forest", "ARIMA", "Naive (last value)"])

# ---------- Train/test split based on chosen horizon ----------
train = df_model.iloc[:-horizon]
test = df_model.iloc[-horizon:]

def run_naive(train, test):
    y_pred = test['lag1']
    y_pred.index = test['Date']
    return y_pred

def run_arima(train, test):
    model = ARIMA(train[target], order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    return pd.Series(forecast.values, index=test['Date'])

def run_rf(train, test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train[target])
    y_pred = model.predict(test[features])
    return pd.Series(y_pred, index=test['Date'])

if model_choice == "Naive (last value)":
    y_pred = run_naive(train, test)
elif model_choice == "ARIMA":
    y_pred = run_arima(train, test)
else:
    y_pred = run_rf(train, test)

y_true = test.set_index('Date')[target]
mae = mean_absolute_error(y_true, y_pred)

st.subheader(f"Forecast results \u2014 {model_choice}, {horizon}-day horizon")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"{mae:.1f}")
col2.metric("Test window", f"{test['Date'].min().date()} to {test['Date'].max().date()}")

comparison_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
st.line_chart(comparison_df)

st.divider()

# ---------- Scenario comparison view ----------
st.subheader("Scenario comparison \u2014 all models side by side")
results = {}
for name, func in [("Naive", run_naive), ("ARIMA", run_arima), ("Random Forest", run_rf)]:
    pred = func(train, test)
    results[name] = mean_absolute_error(y_true, pred)

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MAE'])
st.bar_chart(results_df)
st.dataframe(results_df.style.format({"MAE": "{:.1f}"}))