"""REST API.  Run: uvicorn api:app --reload   (docs at /docs)"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PACKAGING, PRODUCTS
from src.predict import predict_one

app = FastAPI(title="Food Spoilage Prediction API", version="1.0.0")


class Shipment(BaseModel):
    product: str = Field(..., examples=["Milk"])
    packaging: str = Field("Standard", examples=["Vacuum"])
    avg_temp_c: float
    temp_std_c: float = 1.0
    humidity_pct: float = 80
    transit_hours: float
    days_since_harvest: float
    co2_ppm: float = 600
    ethylene_ppm: float = 5
    door_open_events: int = 3
    vibration_g: float = 0.4


@app.get("/")
def root():
    return {"status": "ok", "products": list(PRODUCTS), "packaging": list(PACKAGING)}


@app.post("/predict")
def predict(shipment: Shipment):
    if shipment.product not in PRODUCTS:
        raise HTTPException(400, f"Unknown product. Choose from {list(PRODUCTS)}")
    try:
        return predict_one(shipment.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
