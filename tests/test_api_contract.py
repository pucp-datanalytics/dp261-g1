"""Contract tests for Sprint 6 API.

These tests are intended to run after `dvc pull` has materialized the model.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_version_endpoint():
    client = TestClient(app)
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"]
    assert body["threshold"] == 0.70


def test_predict_contract_if_model_available():
    if not Path("models/final_model.pkl").exists():
        # In CI without DVC remote, this test is intentionally skipped.
        return
    payload = json.loads(Path("handoff/contracts/example_request.json").read_text(encoding="utf-8"))
    client = TestClient(app)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    for key in ["risk_score", "threshold", "prediction", "decision", "model_version"]:
        assert key in body
    assert body["threshold"] == 0.70
