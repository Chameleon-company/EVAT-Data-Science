"""
EV Usage Simulation, Prediction & Optimal Pricing
-------------------------------------------------
1. Simulates 6 months of daily EV charger usage across suburbs.
2. Trains a neural network to predict usage based on features.
3. Estimates revenue-maximizing optimal prices per suburb.
"""

# ===============================
# Imports
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# Step 1: Simulate Usage Data
# ===============================
# Load clustered dataset
clustered_df = pd.read_csv("clustered_suburbs (1).csv")

# Generate 6 months of daily usage per suburb
dates = pd.date_range("2025-01-01", "2025-06-30")
simulated_data = []

for _, row in clustered_df.iterrows():
    # Base demand by cluster
    if "Urban EV-Ready" in row['Cluster_Label']:
        base_demand = 60
    elif "Growth Potential" in row['Cluster_Label']:
        base_demand = 30
    elif "Infrastructure Gap" in row['Cluster_Label']:
        base_demand = 10
    else:
        base_demand = 40

    for d in dates:
        season_factor = 1.2 if d.month in [1, 2] else (0.9 if d.month == 6 else 1.0)
        weekend_factor = 0.8 if d.weekday() >= 5 else 1.0
        price = np.random.uniform(5, 15)

        mean_usage = base_demand * season_factor * weekend_factor * (10 / price)
        usage = max(0, np.random.poisson(mean_usage))

        simulated_data.append([row['Suburb'], row['Cluster'], d, price, usage])

# Final simulated dataset
usage_df = pd.DataFrame(simulated_data, columns=['Suburb', 'Cluster', 'Date', 'Price', 'Usage'])
usage_df.to_csv("simulated_rental_usage.csv", index=False)

# ===============================
# Step 2: Preprocessing
# ===============================
df = usage_df.copy()

# One-hot encode suburb
df = pd.get_dummies(df, columns=['Suburb'], drop_first=True)

# Features & target
X = df.drop(['Usage', 'Date'], axis=1, errors='ignore')
y = df['Usage']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# ===============================
# Step 3: Dataset & DataLoader
# ===============================
class EVUsageDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(EVUsageDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(EVUsageDataset(X_test_tensor, y_test_tensor), batch_size=64)

# ===============================
# Step 4: Neural Network Model
# ===============================
class EVNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = EVNet(input_dim=X.shape[1])

# ===============================
# Step 5: Training Setup
# ===============================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
epochs, patience = 50, 10

train_losses, test_losses = [], []
best_loss, counter = float('inf'), 0

# ===============================
# Step 6: Training Loop
# ===============================
for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test_tensor), y_test_tensor).item()
        test_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_loss:
        best_loss, counter, best_model_state = val_loss, 0, model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# ===============================
# Step 7: Results & Evaluation
# ===============================
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.show()

# Final evaluation
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    y_pred_final = model(X_test_tensor).numpy()

mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
print(f"\nFinal Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# ===============================
# Step 8: Optimal Pricing Function
# ===============================
def optimal_price_nn(suburb, cluster, lag_usage, model, scaler, feature_columns):
    """Finds the revenue-maximizing price for a given suburb."""
    prices, revenues = np.linspace(5, 10, 100), []
    for p in prices:
        input_dict = {col: 0 for col in feature_columns}
        input_dict.update({"Price": p, "Cluster": cluster, "Lag_Usage": lag_usage})
        suburb_col = f"Suburb_{suburb}"
        if suburb_col in input_dict:
            input_dict[suburb_col] = 1

        input_df = pd.DataFrame([[input_dict[col] for col in feature_columns]], columns=feature_columns)
        X_input = torch.tensor(scaler.transform(input_df), dtype=torch.float32)

        with torch.no_grad():
            usage_pred = model(X_input).item()
        revenues.append(p * usage_pred)
    return float(prices[np.argmax(revenues)])

# ===============================
# Step 9: Optimal Prices per Suburb
# ===============================
optimal_prices = []
for suburb in clustered_df['Suburb'].unique():
    cluster_val = clustered_df.loc[clustered_df['Suburb'] == suburb, 'Cluster'].iloc[0]
    lag_usage_val = usage_df.loc[usage_df['Suburb'] == suburb, 'Usage'].mean()
    price = optimal_price_nn(suburb, cluster_val, lag_usage_val, model, scaler, list(X.columns))
    optimal_prices.append([suburb, cluster_val, round(lag_usage_val, 2), round(price, 2)])

optimal_price_df = pd.DataFrame(optimal_prices, columns=['Suburb', 'Cluster', 'Lag_Usage', 'Optimal_Price'])
optimal_price_df.to_csv("optimal_prices_all_suburbs.csv", index=False)

print("\nSample Optimal Prices:")
print(optimal_price_df.head())
