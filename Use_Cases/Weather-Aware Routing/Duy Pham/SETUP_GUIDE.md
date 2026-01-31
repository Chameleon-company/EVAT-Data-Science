# Server Setup Guide

## Issue Diagnosis

The server cannot start because required Python packages are missing. Ensure all dependencies are properly installed.

## Solution Steps

### 1. Check Python Environment
```bash
cd "/Users/wingyip/Desktop/EVAT-Data-Science/Use_Cases/Weather-Aware Routing/Duy Pham"
python3 --version
which python3
```

### 2. Install All Required Packages
```bash
pip3 install pandas scikit-learn shapely geopandas flask flask-cors joblib

# Or if using virtual environment
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn shapely geopandas flask flask-cors joblib
```

### 3. Verify Installation
```bash
python3 -c "import pandas, sklearn, shapely, geopandas, flask, joblib; print('All packages installed!')"
```

### 4. Start Server
```bash
python3 api.py
```

You should see:
```
==================================================
Starting EV Traffic & Weather Model API...
Server will be available at http://localhost:5001
==================================================
 * Running on http://0.0.0.0:5001
```

### 5. Test API

In another terminal window:

**Test root endpoint:**
```bash
curl http://localhost:5001/
```

**Test prediction endpoint:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2023, "start_lat": -37.8136, "start_lon": 144.9631}'
```

## Common Issues

### Issue: ModuleNotFoundError
**Solution:** Ensure all packages are installed in the correct Python environment

### Issue: Port 5001 Already in Use
**Solution:** Modify the port number in the last line of `api.py`, for example change to 5002

### Issue: CORS Error (Frontend Cannot Connect)
**Solution:** Install flask-cors: `pip3 install flask-cors`

## Next Steps

1. ✅ Backend code fixed (completed)
2. ✅ Frontend TypeScript client created (completed)
3. ⏳ Install dependencies (manual step required)
4. ⏳ Start server (manual step required)
5. ⏳ Test API connection (manual step required)
