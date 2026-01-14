from fastapi import FastAPI
from .schema import CostRequest, CostResponse
from .inference import predict_cost

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=CostResponse)
def predict(req: CostRequest):
    y = predict_cost(req.dict())
    return CostResponse(predicted_savings=y)