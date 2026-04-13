"""
api/app.py  —  A-DIDS Production API
Exposes the drone IDS as a high-performance microservice using FastAPI.
Includes endpoints for real-time prediction and system metrics.
"""

import os
import sys
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_PATH, FEATURES

app = FastAPI(
    title="A-DIDS Production API",
    description="Tactical Drone Intrusion Detection System Microservice",
    version="1.0.0"
)

# Global Model Cache
MODEL = None
STATS = {
    "requests": 0,
    "attacks_detected": 0,
    "avg_latency_ms": 0.0
}

class FlowData(BaseModel):
    # Defining the core 19 features required for prediction
    features: List[float]

@app.on_event("startup")
def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
    else:
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"status": "online", "system": "A-DIDS Tactical IDS"}

@app.post("/predict")
def predict_flow(data: FlowData):
    global MODEL
    if len(data.features) != len(FEATURES):
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {len(FEATURES)} features, got {len(data.features)}"
        )
    
    start_time = time.time()
    
    # Convert to DataFrame
    df = pd.DataFrame([data.features], columns=FEATURES)
    
    # Prediction
    pred = int(MODEL.predict(df)[0])
    prob = float(MODEL.predict_proba(df)[0].max())
    latency = (time.time() - start_time) * 1000
    
    # Update Stats
    STATS["requests"] += 1
    if pred == 1:
        STATS["attacks_detected"] += 1
    # Simple moving average for latency
    STATS["avg_latency_ms"] = (STATS["avg_latency_ms"] * (STATS["requests"] - 1) + latency) / STATS["requests"]
    
    return {
        "prediction": "ATTACK" if pred == 1 else "BENIGN",
        "class_id": pred,
        "confidence": round(prob, 4),
        "latency_ms": round(latency, 4)
    }

@app.get("/metrics")
def get_metrics():
    return {
        "uptime_status": "healthy",
        "total_requests": STATS["requests"],
        "attacks_detected": STATS["attacks_detected"],
        "avg_inference_latency_ms": round(STATS["avg_latency_ms"], 4),
        "model_version": "XGBoost-v1.0-ISOT"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
