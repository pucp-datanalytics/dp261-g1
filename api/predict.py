"""Model loading, prediction and explainability utilities for Sprint 6."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from api.settings import PROJECT_ROOT, settings
from api.schemas import BusinessValueAssumptions, PredictionResponse, ShapFeature

# Make project modules importable when the API runs from Docker or repo root.
sys.path.append(str(PROJECT_ROOT))

try:
    from src.config import TARGET
except Exception:  # pragma: no cover - defensive fallback
    TARGET = "IsBadBuy"

try:
    from src.preprocessing import prepare_features
except Exception:  # pragma: no cover - defensive fallback
    prepare_features = None

log = logging.getLogger("api.predict")

DEFAULT_BUSINESS_ASSUMPTIONS = {
    "benefit_tp": 2500,
    "cost_fp": -900,
    "cost_fn": 0,
    "benefit_tn": 0,
}


@lru_cache(maxsize=1)
def load_metadata() -> Dict[str, Any]:
    if settings.metadata_path.exists():
        with open(settings.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_name": "LogisticRegression",
        "threshold": settings.threshold,
        "business_assumptions": DEFAULT_BUSINESS_ASSUMPTIONS,
    }


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {settings.model_path}. Ejecuta `dvc pull` antes de levantar la API."
        )
    return joblib.load(settings.model_path)


def model_sha() -> str:
    if not settings.model_path.exists():
        return "model-not-found"
    h = hashlib.sha256()
    with open(settings.model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop target if accidentally sent and normalize numpy/pandas missing values."""
    clean = {k: (None if pd.isna(v) else v) for k, v in payload.items() if k != TARGET}
    return clean


def payload_to_features(payload: Dict[str, Any]) -> pd.DataFrame:
    raw = pd.DataFrame([sanitize_payload(payload)])
    if prepare_features is None:
        return raw
    try:
        return prepare_features(raw)
    except Exception as exc:
        # This fallback keeps the error message clearer during early Sprint 6
        # integration, but the expected production path is prepare_features(raw).
        log.warning("prepare_features failed; using raw payload. error=%s", exc)
        return raw


def score_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.asarray(proba).ndim == 2:
            return np.asarray(proba)[:, 1].astype(float)
        return np.asarray(proba).astype(float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).astype(float)
        # Logistic transform to keep output in [0, 1].
        return 1 / (1 + np.exp(-scores))
    preds = np.asarray(model.predict(X)).astype(float)
    return preds


def _get_pipeline_parts(model: Any) -> Tuple[Any, Any]:
    if not hasattr(model, "named_steps"):
        return None, None
    return model.named_steps.get("preprocess"), model.named_steps.get("clf")


def _feature_names(preprocess: Any, X: pd.DataFrame) -> Iterable[str]:
    if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
        return preprocess.get_feature_names_out()
    return X.columns


def _to_dense(row: Any) -> np.ndarray:
    if hasattr(row, "toarray"):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


def explain_prediction(model: Any, X: pd.DataFrame, top_n: int = 8) -> Tuple[List[ShapFeature], str]:
    """Return SHAP top features when available, with a coefficient fallback.

    For the expected LogisticRegression pipeline, SHAP LinearExplainer is used
    with a small background sample if available. If SHAP cannot run in the
    deployment environment, the API returns coefficient contributions instead
    and marks the explanation method accordingly.
    """
    preprocess, clf = _get_pipeline_parts(model)
    if preprocess is None or clf is None:
        return [], "unavailable: model is not a sklearn Pipeline with preprocess + clf"

    try:
        row_transformed = preprocess.transform(X)
        names = list(_feature_names(preprocess, X))
    except Exception as exc:
        return [], f"unavailable: preprocessing transform failed ({exc})"

    # Preferred: SHAP for linear models.
    try:
        import shap  # type: ignore

        background = row_transformed
        if settings.shap_background_path.exists() and prepare_features is not None:
            bg_raw = pd.read_csv(settings.shap_background_path).drop(columns=[TARGET], errors="ignore")
            bg_raw = bg_raw.head(settings.shap_max_background_rows)
            bg_features = prepare_features(bg_raw)
            background = preprocess.transform(bg_features)
        explainer = shap.LinearExplainer(clf, background)
        values = explainer(row_transformed).values
        shap_values = np.asarray(values).ravel()
        top_idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
        return [
            ShapFeature(
                feature=str(names[i]),
                shap_value=float(shap_values[i]),
                impact_direction="sube_riesgo" if shap_values[i] >= 0 else "baja_riesgo",
            )
            for i in top_idx
        ], "shap.LinearExplainer"
    except Exception as exc:
        log.warning("SHAP failed; falling back to linear contributions. error=%s", exc)

    # Fallback: linear contribution = transformed value * coefficient.
    try:
        if not hasattr(clf, "coef_"):
            return [], "unavailable: classifier has no coefficients and SHAP failed"
        row_values = _to_dense(row_transformed[0])
        coefs = np.asarray(clf.coef_).ravel()
        contributions = row_values * coefs
        top_idx = np.argsort(np.abs(contributions))[::-1][:top_n]
        return [
            ShapFeature(
                feature=str(names[i]),
                shap_value=float(contributions[i]),
                impact_direction="sube_riesgo" if contributions[i] >= 0 else "baja_riesgo",
            )
            for i in top_idx
        ], "linear_contributions_fallback"
    except Exception as exc:
        return [], f"unavailable: fallback failed ({exc})"


def predict_one(payload: Dict[str, Any]) -> PredictionResponse:
    metadata = load_metadata()
    threshold = float(metadata.get("threshold", settings.threshold))
    model_name = str(metadata.get("model_name", "LogisticRegression"))
    assumptions = metadata.get("business_assumptions", DEFAULT_BUSINESS_ASSUMPTIONS)

    model = load_model()
    X = payload_to_features(payload)
    score = float(score_model(model, X)[0])
    prediction = int(score >= threshold)
    decision = "Revisar / rechazar como posible Bad Buy" if prediction == 1 else "Continuar evaluación"
    shap_features, explanation_method = explain_prediction(model, X)

    return PredictionResponse(
        risk_score=score,
        threshold=threshold,
        prediction=prediction,
        decision=decision,
        model_name=model_name,
        model_version=settings.model_version,
        api_version=settings.api_version,
        business_value_assumptions=BusinessValueAssumptions(**assumptions),
        shap_top_features=shap_features,
        explanation_method=explanation_method,
    )
