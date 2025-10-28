#To Run: from the project main root folder use command "pytest -v"

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test if the API root works."""
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_endpoint():
    with open("tests/sample.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "confidence" in data
    assert data["class"] in ["with_mask", "without_mask"]  # optional extra validation
    assert 0 <= data["confidence"] <= 1

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 400 or response.status_code == 422


