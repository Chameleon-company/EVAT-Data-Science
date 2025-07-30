import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset

df = pd.read_csv("ml_ev_charging_dataset.csv")

# Drop missing or null values 
df.dropna(inplace=True)

# Encode categorical variable (Station_Name)
le = LabelEncoder()
df['Station_Encoded'] = le.fit_transform(df['Station_Name'])

#################REGRESSION MODEL#######################

# derive haversine distance as a feature
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

df['Geo_Distance'] = haversine(df['Latitude'], df['Longitude'],
                               df['Suburb_Location_Lat'], df['Suburb_Location_Lon'])


# Predict ETA_min using coordinates and encoded station

features_reg = ['Longitude', 'Latitude', 'Suburb_Location_Lat',
                'Suburb_Location_Lon', 'Geo_Distance', 'Station_Encoded']
X_reg = df[features_reg]
y_reg = df['ETA_min']

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_pred_lr = lr.predict(X_test_reg)

# Model 2: Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_reg, y_train_reg)
y_pred_rf = rf.predict(X_test_reg)

# Model 3: XGBoost Regressor
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train_reg, y_train_reg)
y_pred_xgb = xgb.predict(X_test_reg)

# Evaluation (Regression)
def regression_results(y_true, y_pred, model_name):
    print(f"\n{model_name} Regression Results:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

regression_results(y_test_reg, y_pred_lr, "Linear")
regression_results(y_test_reg, y_pred_rf, "Random Forest")
regression_results(y_test_reg, y_pred_xgb, "XGBoost")


