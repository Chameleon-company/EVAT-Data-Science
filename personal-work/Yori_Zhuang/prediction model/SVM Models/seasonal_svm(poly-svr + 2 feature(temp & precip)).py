import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

evcs_df = pd.read_csv("EVCS_Usage_Three_Years.csv")
weather_df = pd.read_csv("weather 2016-09-01 to 2019-05-28.csv")

evcs_df['Start Date'] = pd.to_datetime(evcs_df['Start Date'])
daily_usage = evcs_df.groupby(evcs_df['Start Date'].dt.date)['Total kWh'].sum().reset_index()
daily_usage.columns = ['date', 'total_kwh']
daily_usage['date'] = pd.to_datetime(daily_usage['date'])

weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df.rename(columns={'datetime': 'date'}, inplace=True)

merged_df = pd.merge(daily_usage, weather_df, on='date', how='inner')

def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

merged_df['month'] = merged_df['date'].dt.month
merged_df['season'] = merged_df['month'].apply(get_season)

# Initialize the result container
season_models_results = {}

plt.figure(figsize=(16, 12))

# Loop through each season for modeling and plotting
for idx, season in enumerate(['Spring', 'Summer', 'Autumn', 'Winter']):
    season_data = merged_df[merged_df['season'] == season].copy()
    season_data = season_data[['temp', 'humidity', 'precip', 'total_kwh']].dropna()

    season_data = season_data[
        (season_data['temp'] >= -50) &
        (season_data['humidity'] >= 0) &
        (season_data['precip'] >= 0) &
        (season_data['total_kwh'] >= 0)
        ]

    X = season_data[['temp', 'precip']]
    y = season_data['total_kwh']

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

    svr = SVR(kernel='poly')
    svr.fit(X_train, y_train)

    y_pred_scaled = svr.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    season_models_results[season] = {'RMSE': round(rmse, 2), 'R2': round(r2, 4)}

    # Prediction vs Actual
    plt.subplot(2, 2, idx + 1)
    plt.scatter(y_test_orig, y_pred, alpha=0.6, color='teal')
    plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], 'r--')
    plt.title(f'{season}: Predicted vs Actual\nRMSE={rmse:.2f}, R²={r2:.4f}')
    plt.xlabel('Actual Charging (kWh)')
    plt.ylabel('Predicted Charging (kWh)')
    plt.grid(True)

plt.tight_layout()
plt.show()

print("Seasonal SVM Model Results:")
for season, result in season_models_results.items():
    print(f"{season}: RMSE = {result['RMSE']}, R² = {result['R2']}")
