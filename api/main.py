"""FastAPI service for Bad Buy risk prediction."""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.predict import load_metadata, load_model, model_sha, predict_one
from api.schemas import BadBuyFeatures, BatchPredictionRequest, BatchPredictionResponse, HealthResponse, PredictionResponse, VersionResponse
from api.settings import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

app = FastAPI(title="DP261 Bad Buy Prediction API", version=settings.api_version)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def log_event(event: str, **kwargs) -> None:
    log.info(json.dumps({"event": event, "ts": time.time(), **kwargs}, ensure_ascii=False, default=str))


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not settings.require_api_key:
        return
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API key no configurada")
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="API key inválida")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log_event("request", path=request.url.path, method=request.method, status_code=response.status_code, latency_ms=round((time.time() - t0) * 1000, 2))
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True, model_path=str(settings.model_path))
    except Exception as exc:
        return HealthResponse(status=f"error: {exc}", model_loaded=False, model_path=str(settings.model_path))


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    metadata = load_metadata()
    return VersionResponse(api_version=settings.api_version, model_version=str(metadata.get("model_version", settings.model_version)), model_name=str(metadata.get("model_name", "unknown")), model_sha=model_sha(), threshold=float(metadata.get("threshold", 0.5)))


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(features: BadBuyFeatures) -> PredictionResponse:
    try:
        return predict_one(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log_event("prediction_error", error=str(exc))
        raise HTTPException(status_code=400, detail=f"No se pudo generar la predicción: {exc}") from exc


@app.post("/predict_batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = [predict_one(record.model_dump()) for record in payload.records]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)
