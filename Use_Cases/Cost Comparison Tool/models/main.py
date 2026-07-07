from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schema import CostRequest, CostResponse
from .inference import predict_cost

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=CostResponse)
def predict(req: CostRequest):
    y = predict_cost(req.dict())
    return CostResponse(predicted_savings=y)