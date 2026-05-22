"""Smoke test for local or deployed API.

Usage:
    python api/smoke_test.py --url http://localhost:8000
    API_KEY=demo python api/smoke_test.py --url https://api.example.com
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--payload", default="handoff/contracts/example_request.json")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    headers = {}
    if os.getenv("API_KEY"):
        headers["x-api-key"] = os.getenv("API_KEY")

    health = requests.get(f"{base_url}/health", timeout=10)
    print("GET /health", health.status_code, health.text)
    health.raise_for_status()

    version = requests.get(f"{base_url}/version", timeout=10)
    print("GET /version", version.status_code, version.text)
    version.raise_for_status()

    with open(Path(args.payload), "r", encoding="utf-8") as f:
        payload = json.load(f)

    pred = requests.post(f"{base_url}/predict", json=payload, headers=headers, timeout=20)
    print("POST /predict", pred.status_code, pred.text)
    pred.raise_for_status()

    body = pred.json()
    required = {"risk_score", "threshold", "prediction", "decision", "model_version"}
    missing = sorted(required - body.keys())
    if missing:
        raise AssertionError(f"Missing response keys: {missing}")
    print("Smoke test OK")


if __name__ == "__main__":
    main()
