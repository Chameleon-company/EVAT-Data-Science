import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. merge datasets
evcs_df = pd.read_csv("EVCS_Usage_Three_Years.csv")
weather_df = pd.read_csv("weather 2016-09-01 to 2019-05-28.csv")
evcs_df['Start Date'] = pd.to_datetime(evcs_df['Start Date'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df.rename(columns={'datetime': 'date'}, inplace=True)

# 2. counts daily charging amount
daily_counts = evcs_df.groupby(evcs_df['Start Date'].dt.date).size().reset_index(name='charge_count')
daily_counts['date'] = pd.to_datetime(daily_counts['Start Date'])
daily_counts.drop(columns=['Start Date'], inplace=True)

# 3. merge 2 datasets by date
merged_df = pd.merge(daily_counts, weather_df, on='date', how='inner')

# 4. adding season factors
merged_df['month'] = merged_df['date'].dt.month
def get_season(month):
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'
merged_df['season'] = merged_df['month'].apply(get_season)

# 5. data wrangling
df = merged_df[['charge_count', 'temp', 'humidity', 'precip', 'season']].dropna()
df = df[(df['charge_count'] >= 0) & (df['temp'] >= -50) & (df['humidity'] >= 0) & (df['precip'] >= 0)]

# 6. Standardization
features = ['charge_count', 'temp', 'humidity', 'precip']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# 7. label_encoder
label_encoder = LabelEncoder()
season_labels = label_encoder.fit_transform(df['season'])
season_onehot = to_categorical(season_labels)

# 8. Construct sequence data (time window length is 7 days)
def create_sequences(X, y, window_size=7):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(scaled_features, season_onehot, window_size=7)

# 9. Divide the training set and test set
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# 10. RNN
model = Sequential()
model.add(SimpleRNN(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# 11. visualization
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('RNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 12. Test Accuracy
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")
