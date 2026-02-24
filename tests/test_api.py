"""
Tests for Customer Churn Prediction API
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path so app can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)

# -----------------------------------------------
# SAMPLE PAYLOADS
# -----------------------------------------------

VALID_PAYLOAD = {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 100000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0,
    "Satisfaction_Score": 3,
    "Point_Earned": 500,
    "Geography_Germany": 0,
    "Geography_Spain": 0,
    "Gender_Male": 1,
    "Card_Type_GOLD": 1,
    "Card_Type_PLATINUM": 0,
    "Card_Type_SILVER": 0
}

CHURN_RISK_PAYLOAD = {
    "CreditScore": 350,
    "Age": 55,
    "Tenure": 1,
    "Balance": 200000.0,
    "NumOfProducts": 1,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 30000.0,
    "Satisfaction_Score": 1,
    "Point_Earned": 50,
    "Geography_Germany": 1,
    "Geography_Spain": 0,
    "Gender_Male": 0,
    "Card_Type_GOLD": 0,
    "Card_Type_PLATINUM": 0,
    "Card_Type_SILVER": 1
}

# -----------------------------------------------
# HOME ROUTE TESTS
# -----------------------------------------------

def test_home_returns_200():
    response = client.get("/")
    assert response.status_code == 200

def test_home_contains_expected_keys():
    response = client.get("/")
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "model" in data
    assert data["status"] == "running"

# -----------------------------------------------
# HEALTH ROUTE TESTS
# -----------------------------------------------

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_model_loaded():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["features_loaded"] is True

# -----------------------------------------------
# MODEL INFO ROUTE TESTS
# -----------------------------------------------

def test_model_info_returns_200():
    response = client.get("/model-info")
    assert response.status_code == 200

def test_model_info_contains_metrics():
    response = client.get("/model-info")
    data = response.json()
    assert "model_type" in data
    assert "test_accuracy" in data
    assert "feature_names" in data
    assert data["model_type"] == "RandomForestClassifier"
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) == 16

# -----------------------------------------------
# PREDICT ROUTE TESTS
# -----------------------------------------------

def test_predict_returns_200_with_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

def test_predict_response_has_required_fields():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert "churn_prediction" in data
    assert "result" in data
    assert "churn_probability" in data
    assert "stay_probability" in data
    assert "confidence" in data

def test_predict_churn_prediction_is_binary():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["churn_prediction"] in [0, 1]

def test_predict_probabilities_sum_to_one():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    total = round(data["churn_probability"] + data["stay_probability"], 2)
    assert total == 1.0

def test_predict_probabilities_are_in_range():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert 0.0 <= data["stay_probability"] <= 1.0

def test_predict_result_matches_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    if data["churn_prediction"] == 1:
        assert data["result"] == "Will Churn"
    else:
        assert data["result"] == "Will Stay"

def test_predict_confidence_is_valid():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["confidence"] in ["High", "Medium", "Low"]

def test_predict_with_churn_risk_profile():
    response = client.post("/predict", json=CHURN_RISK_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in [0, 1]

def test_predict_germany_customer():
    payload = VALID_PAYLOAD.copy()
    payload["Geography_Germany"] = 1
    payload["Geography_Spain"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_predict_spain_customer():
    payload = VALID_PAYLOAD.copy()
    payload["Geography_Germany"] = 0
    payload["Geography_Spain"] = 1
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_predict_france_customer():
    payload = VALID_PAYLOAD.copy()
    payload["Geography_Germany"] = 0
    payload["Geography_Spain"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_predict_all_card_types():
    for gold, plat, silver in [(1,0,0), (0,1,0), (0,0,1), (0,0,0)]:
        payload = VALID_PAYLOAD.copy()
        payload["Card_Type_GOLD"] = gold
        payload["Card_Type_PLATINUM"] = plat
        payload["Card_Type_SILVER"] = silver
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

# -----------------------------------------------
# VALIDATION TESTS (Invalid Input)
# -----------------------------------------------

def test_predict_missing_field_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload["CreditScore"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_age_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["Age"] = 5  # Below minimum (18)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_credit_score_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["CreditScore"] = 999  # Above max (900)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_negative_balance_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["Balance"] = -100.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_num_products_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["NumOfProducts"] = 5  # Max is 4
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_empty_body_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422
