import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ready():
    r = client.get("/ready")
    assert r.status_code == 200

def test_classes():
    r = client.get("/api/v1/classes")
    assert r.status_code == 200
    assert "Normal" in r.json()["classes"]
    assert "Pneumonia" in r.json()["classes"]
    assert "COVID19" in r.json()["classes"]

def test_predict_no_file():
    r = client.post("/api/v1/predict")
    assert r.status_code == 422

def test_feedback():
    r = client.post("/api/v1/feedback", json={
        "prediction_id": "test_001",
        "predicted_class": "Pneumonia",
        "radiologist_confirmed": True,
        "comments": "Correct"
    })
    assert r.status_code == 200
    assert r.json()["status"] == "success"

def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
