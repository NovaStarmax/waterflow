import pytest
from pathlib import Path
from front.model import WaterInput, MODEL_PATH

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="modèle absent")
def test_prediction_stable():
    water = WaterInput(
        ph=7.2, hardness=150.0, solids=350.0, chloramines=2.5,
        sulfate=200.0, conductivity=420.0, organic_carbon=2.5,
        trihalomethanes=40.0, turbidity=2.0
    )
    from front.model import predict
    result = predict(water)

    assert result.potability == 0
    assert 0.4 <= result.probability <= 0.55