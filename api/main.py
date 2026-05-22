"""FastAPI service for Sprint 6 MVP deployment.

Endpoints required by Sprint 6:
- GET /health
- GET /version
- POST /predict
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.predict import load_model, model_sha, predict_one
from api.schemas import (
    BadBuyFeatures,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    VersionResponse,
)
from api.settings import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Bad Buy Prediction API",
    version=settings.api_version,
    description="Sprint 6 MVP API for vehicle Bad Buy risk prediction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def log_event(event: str, **kwargs) -> None:
    payload = {"event": event, "ts": time.time(), **kwargs}
    log.info(json.dumps(payload, ensure_ascii=False, default=str))


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Require x-api-key only when REQUIRE_API_KEY=true.

    Local development can run without a key. In AWS, set REQUIRE_API_KEY=true
    and API_KEY as a GitHub/AWS secret.
    """
    if not settings.require_api_key:
        return
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API key no configurada en el servidor")
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="API key inválida")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        latency_ms = round((time.time() - t0) * 1000, 2)
        log_event(
            "request",
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
            client=request.client.host if request.client else None,
        )
        return response
    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000, 2)
        log_event("request_error", path=request.url.path, method=request.method, latency_ms=latency_ms, error=str(exc))
        raise


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True, model_path=str(settings.model_path))
    except Exception as exc:
        return HealthResponse(status=f"error: {exc}", model_loaded=False, model_path=str(settings.model_path))


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        api_version=settings.api_version,
        model_version=settings.model_version,
        model_sha=model_sha(),
        threshold=settings.threshold,
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(features: BadBuyFeatures) -> PredictionResponse:
    try:
        return predict_one(features.dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log_event("prediction_error", error=str(exc))
        raise HTTPException(status_code=400, detail=f"No se pudo generar la predicción: {exc}") from exc


@app.post("/predict_batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = [predict_one(record.dict()) for record in payload.records]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)
