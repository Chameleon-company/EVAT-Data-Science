from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# 1.Load 2 datasets
charging_df = pd.read_csv("EVCS Usage_Sep16_Aug17_Melbourne.csv")
weather_df = pd.read_csv("1st weather 2016-09-01 to 2017-08-31.csv")

# 2. charger useage data: sum up charging amounts by same date
charging_df['Start Date'] = pd.to_datetime(charging_df['Start Date']).dt.date
daily_charging = charging_df.groupby('Start Date')['Total kWh'].sum().reset_index()
daily_charging.columns = ['date', 'charging_amount']

# 3. weather data: date conversion and temperature conversion
weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.date
weather_df['temperature_c'] = (weather_df['temp'] - 32) * 5.0 / 9.0
daily_weather = weather_df[['datetime', 'temperature_c']]
daily_weather.columns = ['date', 'temperature']

# 4. combine the datasets by "data"
merged_df = pd.merge(daily_charging, daily_weather, on='date')

# 5. Prepare features and target
X = merged_df[['temperature']]  # Temperature as input
Y = merged_df['charging_amount']  # Charging amount as target

# 6. Create polynomial features (degree 2 for quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 7. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)

# 8. Train the polynomial regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Polynomial Regression (Degree 2)")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R-squared (R²) Score:", round(r2, 4))

# 11. Visualization
import numpy as np
X_plot = np.linspace(X.min(), X.max(), 300)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label="Actual", color="lightblue", alpha=0.6)
plt.plot(X_plot, y_plot, label="Polynomial Fit (deg 2)", color="darkorange", linewidth=2)
plt.xlabel("Temperature (°C)")
plt.ylabel("Daily Charging Amount (kWh)")
plt.title("Polynomial Regression: Temperature vs Charging Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
