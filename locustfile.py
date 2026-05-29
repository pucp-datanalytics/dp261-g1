"""Prueba de carga simple para Sprint 6.

Uso:
    locust -f locustfile.py --host http://localhost:8000
"""
from __future__ import annotations

import json
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD = json.loads(Path("handoff/contracts/example_request.json").read_text(encoding="utf-8"))


class MVPUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def predict(self):
        self.client.post("/predict", json=PAYLOAD)

    @task(1)
    def health(self):
        self.client.get("/health")
