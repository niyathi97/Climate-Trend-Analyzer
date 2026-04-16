import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

print("Loading dataset...")

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert date
df['dt'] = pd.to_datetime(df['dt'])

# Rename columns
df.rename(columns={'dt': 'Date', 'LandAverageTemperature': 'Temperature'}, inplace=True)

# Clean data
df = df[['Date', 'Temperature']].dropna()

# Extract year
df['Year'] = df['Date'].dt.year

# Yearly average temperature
yearly_temp = df.groupby('Year')['Temperature'].mean()

# -------------------------------
# 🔮 FORECASTING USING ARIMA
# -------------------------------

# Fit model
model = ARIMA(yearly_temp, order=(2,1,2))
model_fit = model.fit()

# Forecast next 10 years
forecast = model_fit.forecast(steps=10)

# Create future years
last_year = yearly_temp.index[-1]
future_years = [last_year + i for i in range(1, 11)]

# -------------------------------
# 📊 PLOT RESULTS
# -------------------------------

plt.figure()

# Actual data
plt.plot(yearly_temp.index, yearly_temp.values, label="Actual")

# Forecast data
plt.plot(future_years, forecast, label="Forecast", linestyle='dashed')

plt.title("Temperature Forecast (Next 10 Years)")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.legend()

plt.show()