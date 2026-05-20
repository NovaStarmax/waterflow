from fastapi import APIRouter
from api.schemas.water import WaterInput, PredictionOutput
from api.services.predictor import WaterPredictor

router = APIRouter()
predictor = WaterPredictor(model_path="models/model.pkl")  # relatif à la racine du projet

@router.post("/predict", response_model= PredictionOutput)
def predict(water: WaterInput):
    return predictor.predict(water)