"""Basic load test for Sprint 6 API.

Run:
    API_KEY=demo locust -f locustfile.py --host http://localhost:8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD_PATH = Path("handoff/contracts/example_request.json")
with open(PAYLOAD_PATH, "r", encoding="utf-8") as f:
    PAYLOAD = json.load(f)


class BadBuyAPIUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @property
    def headers(self):
        api_key = os.getenv("API_KEY", "")
        return {"x-api-key": api_key} if api_key else {}

    @task(3)
    def predict(self):
        self.client.post("/predict", json=PAYLOAD, headers=self.headers)

    @task(1)
    def health(self):
        self.client.get("/health")
