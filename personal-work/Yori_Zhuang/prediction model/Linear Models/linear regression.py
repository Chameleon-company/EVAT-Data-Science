from sklearn.model_selection import train_test_split
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


# 5. prepare features and target
X = merged_df[['temperature']]  # Temperature as input
Y = merged_df['charging_amount']  # Charging amount as target

# 6. split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 7. create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. make predictions
y_pred = model.predict(X_test)

# 5. evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # coefficient of determination
print("Mean Squared Error (MSE):", round(mse, 2))
print("R-squared (R²) Score:", round(r2, 4))

# 6. Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Actual", color="blue", alpha=0.6)
plt.scatter(X_test, y_pred, label="Predicted", color="red", alpha=0.6)
plt.xlabel("Temperature (°C)")
plt.ylabel("Daily Charging Amount (kWh)")
plt.title("Linear Regression: Temperature vs Predicted Charging Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
