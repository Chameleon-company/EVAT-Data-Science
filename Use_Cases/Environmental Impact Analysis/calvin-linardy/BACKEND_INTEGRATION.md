# Backend Integration Guide — Environmental Impact Analysis

**For: EVAT-App-BE team**
**Use case owner: Calvin Linardy Candra (Data Science)**

The frontend route `/environmental-impact` and wrapper page are already live from Sprint 2.
The Python FastAPI service is built and ready. This guide covers the Node.js backend wiring only.

---

## 1. Start the Python ML service

Before testing, the Python service must be running. Run this from the Data Science repo:

```bash
cd "Use_Cases/Environmental Impact Analysis/calvin-linardy"
pip install -r requirements.txt
python train.py
MONGO_URI="mongodb+srv://ds_team:sWTFhaOYo18t4wrE@cluster0.0neshws.mongodb.net/EVAT" uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload
```

The service runs on **port 8001**. Swagger docs: http://localhost:8001/api/environmental-impact/docs

---

## 2. Three files to update in EVAT-App-BE

Same pattern as Cost Comparison and Demand Forecasting — just add to the existing files.

### predict-service.ts — add one method

```typescript
async getEnvironmentalImpact(payload: object): Promise<any> {
  const response = await fetch("http://localhost:8001/api/environmental-impact/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return await response.json();
}
```

### predict-controller.ts — add one method

```typescript
async getEnvironmentalImpact(req: Request, res: Response): Promise<Response> {
  try {
    const result = await this.predictService.getEnvironmentalImpact(req.body);
    return res.status(200).json(result);
  } catch (error: any) {
    return res.status(500).json({ message: error.message });
  }
}
```

### predict-route.ts — add one route

```typescript
router.post(
  "/environmental-impact",
  authGuard(["user", "admin"]),
  (req, res) => {
    predictController.getEnvironmentalImpact(req, res);
  }
);
```

This exposes the endpoint at: `POST /api/predict/environmental-impact`

---

## 3. Request body to forward to the Python service

The Node backend should forward `req.body` directly to the Python service as-is.

Minimum required fields:

```json
{
  "ev": {
    "make": "Tesla",
    "model": "Model 3",
    "year": 2024
  },
  "ice": {
    "make": "Toyota",
    "model": "RAV4",
    "year": 2024
  },
  "state": "NSW",
  "usage": {
    "annual_km": 13100,
    "driving_style": "mixed",
    "vehicle_age_years": 0
  }
}
```

Valid values:
- `state`: `NSW`, `VIC`, `QLD`, `SA`, `WA`, `TAS`, `ACT`, `NT`
- `driving_style`: `city`, `mixed`, `highway`

---

## 4. Response the Python service returns

```json
{
  "savings": {
    "co2_savings_g_per_km": 21.4,
    "co2_savings_kg_per_year": 280.3,
    "co2_savings_tonnes_lifetime": 4.2,
    "percentage_reduction": 12.3
  },
  "financial": {
    "cost_saving_aud_per_year": 1240.0,
    "cost_saving_aud_lifetime": 18600.0
  },
  "equivalents": {
    "trees_planted_equivalent_per_year": 13,
    "petrol_litres_saved_per_year": 122.5
  },
  "state": "NSW",
  "real_world_adjustment_factor": 1.082
}
```

---

## 5. Vehicle listing endpoints (for frontend dropdowns)

The frontend dropdowns need these — also proxy through Node:

```
GET /api/environmental-impact/vehicles/ev
GET /api/environmental-impact/vehicles/ev/{make}/models
GET /api/environmental-impact/vehicles/ice
GET /api/environmental-impact/vehicles/ice/{make}/models
```

These are GET requests with no body — just forward them to `http://localhost:8001`.

---

## 6. Checklist before marking integration done

- [ ] Python service starts on port 8001 with no errors
- [ ] `POST /api/predict/environmental-impact` returns 200 with JSON (not 401 or 500)
- [ ] Authorization header is included in the frontend fetch (Bearer token)
- [ ] EV and ICE dropdowns populate correctly from vehicle listing endpoints
- [ ] Node backend restarted after adding routes
- [ ] All three services running: React (3000), Node (8080), Python (8001)
- [ ] F12 Network tab shows 200 responses, not empty or 401
