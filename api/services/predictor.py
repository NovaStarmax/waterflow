import joblib
import numpy as np
from api.schemas.water import WaterInput, PredictionOutput

class WaterPredictor:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def _load(self):
        if self.model is None:
            self.model = joblib.load(self.model_path)

    def predict(self, water: WaterInput) -> PredictionOutput:
        self._load()
        features = np.array([[
            water.ph,
            water.hardness,
            water.solids,
            water.chloramines,
            water.sulfate,
            water.conductivity,
            water.organic_carbon,
            water.trihalomethanes,
            water.turbidity
        ]])

        predicted_class = self.model.predict(features)        # retourne [0] ou [1]
        proba = self.model.predict_proba(features)            # retourne [[0.3, 0.7]]

        return PredictionOutput(
            potability= int(predicted_class[0]),
            probability= float(proba[0][1]),
        )