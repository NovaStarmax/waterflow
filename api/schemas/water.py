from pydantic import BaseModel


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
