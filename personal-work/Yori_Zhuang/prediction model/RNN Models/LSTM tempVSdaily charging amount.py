import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

evcs_df = pd.read_csv('EVCS_Usage_Three_Years.csv')
weather_df = pd.read_csv('weather 2016-09-01 to 2019-05-28.csv')
# merge 'Start Date' and 'Start Time'
evcs_df['start_time'] = pd.to_datetime(evcs_df['Start Date'].astype(str) + ' ' + evcs_df['Start Time'].astype(str))
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
evcs_df['date'] = evcs_df['start_time'].dt.date
weather_df['date'] = weather_df['datetime'].dt.date

daily_charge = evcs_df.groupby('date')['Total kWh'].sum().reset_index()
daily_charge.columns = ['date', 'total_kwh']

merged_df = pd.merge(daily_charge, weather_df, on='date')
merged_df = merged_df[['date', 'total_kwh', 'temp']]  # 使用 temp 字段
merged_df = merged_df[(merged_df['total_kwh'] >= 0) & (merged_df['temp'].notnull())]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

merged_df[['temp']] = scaler_x.fit_transform(merged_df[['temp']])
merged_df[['total_kwh']] = scaler_y.fit_transform(merged_df[['total_kwh']])

def create_sequences(data, target, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(target[i + window])
    return np.array(X), np.array(y)

window_size = 7
X, y = create_sequences(merged_df[['temp']].values, merged_df['total_kwh'].values, window_size)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, input_shape=(window_size, 1), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=16,
                    validation_split=0.2, callbacks=[early_stop], verbose=1)

y_pred = model.predict(X_test)
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler_y.inverse_transform(y_pred)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)
print(f"✅ Test RMSE: {rmse:.2f}, R²: {r2:.2f}")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM Prediction vs True Charging Volume')
plt.xlabel('Days')
plt.ylabel('Total kWh')
plt.legend()
plt.tight_layout()
plt.show()
