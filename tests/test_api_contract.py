from fastapi.testclient import TestClient

from api.main import app
from api.predict import load_metadata


def test_health_and_version():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert "model_loaded" in health.json()

    version = client.get("/version")
    assert version.status_code == 200
    assert "model_sha" in version.json()


def test_predict_contract_example():
    import json
    from pathlib import Path

    client = TestClient(app)
    payload = json.loads(Path("handoff/contracts/example_request.json").read_text(encoding="utf-8"))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_segment"] in {"ROJO", "ÁMBAR", "VERDE"}
    assert body["threshold"] == load_metadata()["threshold"]
