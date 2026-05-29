"""Smoke test local para la API.

Ejecutar con la API levantada:
    uvicorn api.main:app --reload
    python api/smoke_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"


def main():
    payload = json.loads(Path("handoff/contracts/example_request.json").read_text(encoding="utf-8"))
    for path in ["/health", "/version"]:
        response = requests.get(f"{BASE_URL}{path}", timeout=10)
        print(path, response.status_code, response.json())
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=20)
    print("/predict", response.status_code, response.json())
    response.raise_for_status()


if __name__ == "__main__":
    main()
