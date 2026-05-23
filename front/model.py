import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from api.schemas.water import WaterInput, PredictionOutput

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"
THRESHOLD = 0.50

# Noms exacts des colonnes utilisés à l'entraînement
FEATURE_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

_model = None


def predict(payload: WaterInput) -> PredictionOutput:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    x = pd.DataFrame([[
        payload.ph, payload.hardness, payload.solids, payload.chloramines,
        payload.sulfate, payload.conductivity, payload.organic_carbon,
        payload.trihalomethanes, payload.turbidity,
    ]], columns=FEATURE_COLUMNS)
    proba = float(_model.predict_proba(x)[0][1])
    return PredictionOutput(potability=int(proba >= THRESHOLD), probability=proba)
