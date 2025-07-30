import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
evcs_df = pd.read_csv("EVCS_Usage_Three_Years.csv")
weather_df = pd.read_csv("weather 2016-09-01 to 2019-05-28.csv")

# Convert date strings to datetime objects
evcs_df['Start Date'] = pd.to_datetime(evcs_df['Start Date'])
daily_usage = evcs_df.groupby(evcs_df['Start Date'].dt.date)['Total kWh'].sum().reset_index()
daily_usage.columns = ['date', 'total_kwh']
daily_usage['date'] = pd.to_datetime(daily_usage['date'])

weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df.rename(columns={'datetime': 'date'}, inplace=True)

# Merge datasets on the date column
merged_df = pd.merge(daily_usage, weather_df, on='date', how='inner')

# Prepare features (temperature) and target (total_kwh)
X = merged_df[['temp']].values
y = merged_df['total_kwh'].values

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train SVM regression model
svm = SVR(kernel='poly', C=10, epsilon=0.1)
svm.fit(X_train, y_train)

# Predict and inverse transform
y_pred_scaled = svm.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
r2 = r2_score(y_test_inv, y_pred)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(scaler_X.inverse_transform(X_test), y_test_inv, color='blue', label='Actual')
plt.scatter(scaler_X.inverse_transform(X_test), y_pred, color='red', alpha=0.6, label='Predicted')
plt.xlabel('Temperature (°C)')
plt.ylabel('Total Charging Amount (kWh)')
plt.title(f'SVM Prediction of Temperature Impact on Charging Amount\nRMSE = {rmse:.2f}, R² = {r2:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
