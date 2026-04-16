import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("🌍 Climate Trend Analyzer")

# Upload file
uploaded_file = st.file_uploader("Upload Climate Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Convert date
    df['dt'] = pd.to_datetime(df['dt'])
    df.rename(columns={'dt': 'Date', 'LandAverageTemperature': 'Temperature'}, inplace=True)

    df = df[['Date', 'Temperature']].dropna()
    df['Year'] = df['Date'].dt.year

    yearly_temp = df.groupby('Year')['Temperature'].mean()

    # 📊 Trend Graph
    st.subheader("📈 Temperature Trend")
    fig1, ax1 = plt.subplots()
    ax1.plot(yearly_temp.index, yearly_temp.values)
    st.pyplot(fig1)

    # ⚠️ Anomaly Detection
    mean = yearly_temp.mean()
    std = yearly_temp.std()

    z_scores = (yearly_temp - mean) / std
    anomalies = yearly_temp[abs(z_scores) > 2]

    st.subheader("⚠️ Anomalies")
    st.write(anomalies)

    # 🔮 Forecast
    model = ARIMA(yearly_temp, order=(2,1,2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)
    future_years = [yearly_temp.index[-1] + i for i in range(1, 11)]

    st.subheader("🔮 Forecast")
    fig2, ax2 = plt.subplots()
    ax2.plot(yearly_temp.index, yearly_temp.values, label="Actual")
    ax2.plot(future_years, forecast, linestyle='dashed', label="Forecast")
    ax2.legend()
    st.pyplot(fig2)