import requests
import json
api = 'http://127.0.0.1:8001'
payload = {
  "Brand": "Toyota", "Model": "Camry", "Year": 2021, "Fuel Type": "Petrol", 
  "Transmission": "Automatic", "Mileage": 50000.0, "car_age": 5, "age_squared": 25,
  "log_mileage": 10.819798, "mileage_per_year": 10000.0, "condition_score": 2}

records = []
for i in range(5):
    p = dict(payload)
    p['car_age'] = 5 + i
    p['age_squared'] = p['car_age'] ** 2
    records.append({"row_id": str(i), "features": p})

r = requests.post(f"{api}/predict/batch", json={"records": records}).json()
for p in r.get("predictions", []):
    print(p["row_id"], p["predicted_price"])
