# To Run: from the project main root folder use command "pytest -v"

import sys, os, pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.main import app

client = TestClient(app)


def assert_success(response):
    """Helper to assert any 2xx success code."""
    assert 200 <= response.status_code < 300, f"Unexpected status: {response.status_code}"


def test_root_endpoint():
    """Test if the API root works."""
    response = client.get("/")
    assert_success(response)


def test_prediction_endpoint():
    """Test the /predict endpoint with a sample image."""
    with open("tests/sample.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", f, "image/jpeg")}
        )

    assert_success(response)

    data = response.json()
    assert "class" in data, "Missing 'class' in response"
    assert "confidence" in data, "Missing 'confidence' in response"
    assert data["class"] in ["with_mask", "without_mask"], "Invalid class label"
    assert 0 <= data["confidence"] <= 1, "Confidence must be between 0 and 1"


def test_predict_no_file():
    """Test /predict with no file sent."""
    response = client.post("/predict")
    # Expecting 400 (Bad Request) or 422 (Unprocessable Entity)
    assert response.status_code in [400, 422], f"Unexpected status: {response.status_code}"
