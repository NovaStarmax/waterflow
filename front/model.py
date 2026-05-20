from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from pydantic import BaseModel

API_URL = os.getenv("WATERFLOW_API", "http://localhost:8000/api/predict")
MODEL_PATH = Path(__file__).parent / "model.pkl"

THRESHOLD = 0.50

# Ordre strict imposé par le modèle (Pipeline entraînée dans cet ordre)
FEATURE_ORDER = [
    "ph", "hardness", "solids", "chloramines", "sulfate",
    "conductivity", "organic_carbon", "trihalomethanes", "turbidity",
]


class WaterInput(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float


class PredictionOutput(BaseModel):
    potability: int
    probability: float


# --- Backends -----------------------------------------------------------------

def _try_api(payload: WaterInput) -> Optional[PredictionOutput]:
    try:
        import requests
        r = requests.post(API_URL, json=payload.model_dump(), timeout=2)
        if r.ok:
            return PredictionOutput(**r.json())
    except Exception:
        return None
    return None


_local_model = None

def _try_local(payload: WaterInput) -> Optional[PredictionOutput]:
    global _local_model
    if not MODEL_PATH.exists():
        return None
    try:
        if _local_model is None:
            _local_model = joblib.load(MODEL_PATH)
        x = np.array([[getattr(payload, f) for f in FEATURE_ORDER]])
        proba = float(_local_model.predict_proba(x)[0][1])
        return PredictionOutput(potability=int(proba >= THRESHOLD), probability=proba)
    except Exception:
        return None


def _mock(payload: WaterInput) -> PredictionOutput:
    score = 1.0
    if not (6.5 <= payload.ph <= 8.5):
        score -= 0.25
    if payload.chloramines > 4:
        score -= 0.20
    if payload.trihalomethanes > 80:
        score -= 0.15
    if payload.turbidity > 5:
        score -= 0.15
    if payload.sulfate > 250:
        score -= 0.10
    if payload.solids > 1000:
        score -= 0.10
    proba = max(0.02, min(0.98, score))
    return PredictionOutput(potability=int(proba >= THRESHOLD), probability=proba)


# --- Public API ---------------------------------------------------------------

def predict(payload: WaterInput) -> tuple[PredictionOutput, str]:
    out = _try_api(payload)
    if out is not None:
        return out, "api"
    out = _try_local(payload)
    if out is not None:
        return out, "local"
    return _mock(payload), "mock"
