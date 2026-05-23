from api.schemas.water import WaterInput, PredictionOutput
from pydantic import ValidationError
import pytest

def test_water_input_valide():
    water = WaterInput(
        ph=7.0, hardness=150.0, solids=350.0,
        chloramines=2.5, sulfate=200.0, conductivity=420.0,
        organic_carbon=2.5, trihalomethanes=40.0, turbidity=1.0
    )
    assert water.ph == 7.0

def test_water_input_rejette_champ_manquant():
    with pytest.raises(ValidationError):
        WaterInput(hardness=150.0)  # ph manquant

def test_prediction_output_valide():
    output = PredictionOutput(potability=1, probability=0.8)
    assert output.potability == 1
    assert output.probability == 0.8

def test_prediction_output_rejette_champ_manquant():
    with pytest.raises(ValidationError):
        PredictionOutput(potability=1)  # probability manquant