import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import settings
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
    r = client.get("/classes")
    print(r.json())
    assert r.status_code == 200
    assert r.json()["classes"] == settings.CLASS_NAMES

def test_predict_no_file():
    r = client.post(f"{settings.API_PREFIX}/predict")
    assert r.status_code == 422

def test_feedback():
    r = client.post(f"{settings.API_PREFIX}/feedback", json={
        "prediction_id": "test_001",
        "predicted_class": settings.CLASS_NAMES[1],
        "radiologist_confirmed": True,
        "comments": "Correct"
    })
    assert r.status_code == 200
    assert r.json()["status"] == "success"

def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
