# EV Congestion Prediction API

A FastAPI-based REST API for real-time EV charging station congestion forecasting using a trained RandomForest model.

## Features

✅ **Real-time Predictions** - Forecast 3-hour arrival counts for charging stations  
✅ **Automatic Feature Engineering** - Fetches and processes external data automatically  
✅ **External Data Integration** - Weather, holidays, events, pedestrian counts  
✅ **Batch Predictions** - Predict for multiple stations in a single request  
✅ **Auto-generated Documentation** - Interactive API docs at `/docs`  
✅ **Health Monitoring** - Health check endpoint for service monitoring  

## Installation

### 1. Install Dependencies
[USE A VENV!](https://docs.python.org/3/library/venv.html)
```bash
# Create VENV
python -m venv .venv
# Bash
source .venv/bin/activate
# Windows CMD
.venv\Scripts\activate.bat
# Install dependencies
pip install -r requirements_api.txt
```

### 2. Ensure Model File Exists

Place your trained `random_forest_model.pkl` in the same directory as `model_api.py`.

### 3. Create and set a `.env` file
Create a `.env` file in the same folder as the `model_api.py`.
```env
BE_SERVER="http://localhost:8080"
ADMIN_AUTH="BE Admin Auth Code Here"
NO_THREADS='5'
```

## Usage

### Start the API Server
Automatically begins predicting and sending full dataset to back-end server.

```bash
# Development mode (auto-reload)
uvicorn model_api:app --reload --port 8000

# Production mode
uvicorn model_api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-07T10:30:00"
}
```

### 2. Model Information
```bash
GET /model/info
```

**Response:**
```json
{
  "model_type": "RandomForestRegressor",
  "n_estimators": 300,
  "max_depth": 15,
  "features_count": 23,
  "features": ["hour", "dayofweek", ...]
}
```

### 3. Single Station Prediction
```bash
POST /predict
Content-Type: application/json

{
  "station_id": "674f97ff3dc8e5d2ac00867a",
  "timestamp": "2026-01-07T14:00:00"  // optional, defaults to now
}
```

**Response:**
```json
{
  "station_id": "674f97ff3dc8e5d2ac00867a",
  "predicted_arrivals": 2.45,
  "timestamp": "2026-01-07T14:00:00",
  "hour": 14,
  "dayofweek": 1,
  "is_weekend": false,
  "is_holiday": false,
  "is_major_event": true,
  "temperature_c": 22.5,
  "precipitation_mm": 0.0,
  "pedestrian_count": 1250.0
}
```

### 4. Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "station_ids": [
    "674f97ff3dc8e5d2ac00867a",
    "674f98013dc8e5d2ac00894a",
    "674f97ff3dc8e5d2ac008456"
  ],
  "timestamp": "2026-01-07T14:00:00"  // optional
}
```

**Response:**
```json
{
  "predictions": [
    {
      "station_id": "674f97ff3dc8e5d2ac00867a",
      "predicted_arrivals": 2.45,
      ...
    },
    {
      "station_id": "674f98013dc8e5d2ac00894a",
      "predicted_arrivals": 1.82,
      ...
    }
  ],
  "count": 3,
  "timestamp": "2026-01-07T14:00:00"
}
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage with Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"station_id": "674f97ff3dc8e5d2ac00867a"}
)
result = response.json()
print(f"Predicted arrivals: {result['predicted_arrivals']:.2f}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "station_ids": [
            "674f97ff3dc8e5d2ac00867a",
            "674f98013dc8e5d2ac00894a"
        ]
    }
)
results = response.json()
for pred in results['predictions']:
    print(f"{pred['station_id']}: {pred['predicted_arrivals']:.2f}")
```

## Example Usage with cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"station_id": "674f97ff3dc8e5d2ac00867a"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "station_ids": [
      "674f97ff3dc8e5d2ac00867a",
      "674f98013dc8e5d2ac00894a"
    ]
  }'
```

## Features Automatically Engineered

The API automatically fetches and engineers the following features:

### Temporal Features
- `hour` - Hour of day (0-23)
- `dayofweek` - Day of week (0=Monday, 6=Sunday)
- `is_weekend` - Weekend indicator

### External Data Features
- **Weather** (from Open-Meteo API)
  - Temperature (max, min, average)
  - Precipitation
  - Wind speed
  
- **Holidays** (Victoria, Australia)
  - Public holiday indicator
  
- **Major Events** (Melbourne-specific)
  - Australian Open
  - AFL Season & Grand Final
  - Melbourne Cup
  - Australian Grand Prix
  - Boxing Day Test
  
- **Pedestrian Counts** (Melbourne pedestrian counting system)
  - Foot traffic for the prediction hour

### Derived Features
- Interaction features (weekend × hour, temperature × precipitation)
- Lag features (set to zero for real-time prediction)

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP POST /predict
       ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
│  ┌───────────────────────────────┐  │
│  │  Feature Engineering Pipeline │  │
│  │  • Temporal features          │  │
│  │  • External data fetching     │  │
│  │  • Feature interactions       │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │    RandomForest Model         │  │
│  │    (300 trees, depth=15)      │  │
│  └───────────────────────────────┘  │
└──────┬──────────────────────────────┘
       │
       │ Prediction Response (JSON)
       ▼
┌─────────────┐
│   Client    │
└─────────────┘
```

## Error Handling

The API includes robust error handling:

- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Prediction or processing failure
- **422 Unprocessable Entity**: Invalid request format

## Logging

The API logs important events:
- Model loading status
- Prediction requests
- External API calls
- Errors and warnings

## Performance Considerations

- **External API Caching**: Consider caching weather/pedestrian data
- **Batch Predictions**: Use batch endpoint for multiple stations
- **Async Operations**: API uses async handlers for concurrent requests
- **Timeouts**: External API calls have 10-second timeouts with fallback defaults

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY model_api.py random_forest_model.pkl ./

EXPOSE 8000

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t ev-prediction-api .
docker run -p 8000:8000 ev-prediction-api
```

### Using systemd (Linux)

Create `/etc/systemd/system/ev-prediction-api.service`:

```ini
[Unit]
Description=EV Congestion Prediction API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ev-prediction-api
ExecStart=/usr/bin/uvicorn model_api:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ev-prediction-api
sudo systemctl start ev-prediction-api
```

## Monitoring

Monitor the API health:

```bash
# Simple health check
watch -n 5 'curl -s http://localhost:8000/health | jq'

# With logging
tail -f /var/log/ev-prediction-api.log
```

## Troubleshooting

### Model Not Loading
- Ensure `random_forest_model.pkl` is in the correct directory
- Check file permissions
- Verify scikit-learn version compatibility

### External API Failures
- The API uses fallback default values when external APIs fail
- Check network connectivity
- Review API rate limits

### Prediction Errors
- Validate input station_id format
- Check timestamp format (ISO 8601)
- Review logs for detailed error messages

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check the API documentation at `/docs`
- Review logs for error details
- Ensure all dependencies are installed
