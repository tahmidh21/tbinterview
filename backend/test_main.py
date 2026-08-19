import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verifies that the root endpoint GET / returns a 200 status code and the expected health check JSON structure."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "message" in data

def test_predict_returns_valid_response():
    """Verifies that POST /predict handles a valid all-zero payload (Low risk profile) and returns the expected JSON structure (risk_label, risk_name, confidence)."""
    payload = {
        "gender": 0,
        "fever_for_two_weeks": 0,
        "coughing_blood": 0,
        "sputum_mixed_with_blood": 0,
        "night_sweats": 0,
        "chest_pain": 0,
        "back_pain_in_certain_parts": 0,
        "shortness_of_breath": 0,
        "weight_loss": 0,
        "body_feels_tired": 0,
        "lumps_that_appear_around_the_armpits_and_neck": 0,
        "cough_and_phlegm_continuously_for_two_weeks_to_four_weeks": 0,
        "swollen_lymph_nodes": 0,
        "loss_of_appetite": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_label" in data
    assert isinstance(data["risk_label"], int)
    assert data["risk_label"] in [0, 1, 2]
    assert "risk_name" in data
    assert data["risk_name"] in ["Low", "Medium", "High"]
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_high_risk_payload():
    """Verifies that POST /predict handles an all-ones payload (High risk profile) and returns a valid response structure."""
    payload = {
        "gender": 1,
        "fever_for_two_weeks": 1,
        "coughing_blood": 1,
        "sputum_mixed_with_blood": 1,
        "night_sweats": 1,
        "chest_pain": 1,
        "back_pain_in_certain_parts": 1,
        "shortness_of_breath": 1,
        "weight_loss": 1,
        "body_feels_tired": 1,
        "lumps_that_appear_around_the_armpits_and_neck": 1,
        "cough_and_phlegm_continuously_for_two_weeks_to_four_weeks": 1,
        "swollen_lymph_nodes": 1,
        "loss_of_appetite": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_label" in data
    assert "risk_name" in data
    assert "confidence" in data

def test_history_returns_list():
    """Verifies that GET /history returns a 200 status code and a list containing recent prediction entries with correct fields."""
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        entry = data[0]
        assert "risk_label" in entry
        assert "risk_name" in entry
        assert "confidence" in entry

def test_feature_importance_valid():
    """Verifies that GET /feature-importance returns a 200 status code, a non-empty list of features with string names and float importances, sorted descending."""
    response = client.get("/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    previous_importance = 1.0
    for item in data:
        assert "feature" in item
        assert isinstance(item["feature"], str)
        assert "importance" in item
        assert isinstance(item["importance"], float)
        # Check that it's sorted descending
        assert item["importance"] <= previous_importance + 1e-6 # small tolerance for floating point
        previous_importance = item["importance"]
