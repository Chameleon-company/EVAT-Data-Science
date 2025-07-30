import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1.
evcs_df = pd.read_csv("EVCS_Usage_Three_Years.csv")
weather_df = pd.read_csv("weather 2016-09-01 to 2019-05-28.csv")
evcs_df['Start Date'] = pd.to_datetime(evcs_df['Start Date'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df.rename(columns={'datetime': 'date'}, inplace=True)

# 2.
daily_counts = evcs_df.groupby(evcs_df['Start Date'].dt.date).size().reset_index(name='charge_count')
daily_counts['date'] = pd.to_datetime(daily_counts['Start Date'])
daily_counts.drop(columns=['Start Date'], inplace=True)

# 3.
merged_df = pd.merge(daily_counts, weather_df, on='date', how='inner')

# 4.
merged_df['month'] = merged_df['date'].dt.month
def get_season(month):
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'
merged_df['season'] = merged_df['month'].apply(get_season)

# 5.
df = merged_df[['charge_count', 'temp', 'season', 'date']]
df = df[(df['charge_count'] >= 0) & (df['temp'].notna())]
df = df.sort_values(by='date').reset_index(drop=True)

# 6. Standardization
scaler = StandardScaler()
df['temp'] = scaler.fit_transform(df[['temp']])
scaled_df = df.copy()

# 7.
def create_sequences(data, window=7):
    X, y, seasons = [], [], []
    for i in range(len(data) - window):
        seq_x = data[['charge_count', 'temp']].iloc[i:i+window].values
        seq_y = data['charge_count'].iloc[i+window]  # y 为原始值
        season = data['season'].iloc[i+window]
        X.append(seq_x)
        y.append(seq_y)
        seasons.append(season)
    return np.array(X), np.array(y), seasons

X, y, seasons = create_sequences(scaled_df, window=7)

# 8.
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
season_test = seasons[split:]

# 9. LSTM
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # output layer predict charge_count

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=0)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Test RMSE: {rmse:.2f}, R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Season': season_test
})

plt.figure(figsize=(8,5))
for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
    subset = results_df[results_df['Season'] == season]
    plt.scatter(subset['Actual'], subset['Predicted'], label=season, alpha=0.6)

#plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.plot([40, 90], [40, 90], 'k--', label='Ideal Line')
plt.xlabel('Actual Charge Count ')
plt.ylabel('Predicted Charge Count ')
plt.title('LSTM Prediction by Season (No Normalization)')
plt.legend()
plt.tight_layout()
plt.show()