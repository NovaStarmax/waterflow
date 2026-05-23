from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_retourne_200():
    response = client.get("/api/health")
    assert response.status_code == 200


def test_predict_retourne_200():
    response = client.post("/api/predict", json={
        "ph": 7.0,
        "hardness": 150.0,
        "solids": 350.0,
        "chloramines": 2.5,
        "sulfate": 200.0,
        "conductivity": 420.0,
        "organic_carbon": 2.5,
        "trihalomethanes": 40.0,
        "turbidity": 1.0
    })
    assert response.status_code == 200


def test_predict_retourne_422():
    response = client.post("/api/predict", json={"hardness": 150.0})
    assert response.status_code == 422
