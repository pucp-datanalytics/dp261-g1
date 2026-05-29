"""Model loading, prediction and explainability utilities.

La API está pensada para que el dashboard reciba una respuesta útil para negocio:
- score de riesgo y decisión de semáforo;
- datos técnicos mínimos del modelo;
- factores explicativos para que el agente entienda por qué revisar o detener una compra.

Nota sobre explicabilidad:
La API intenta devolver explicaciones locales por instancia. Para árboles y boosting se usa
SHAP cuando el estimador lo soporta directamente. Para Bagging sobre árboles se calcula una
aproximación local promediando los valores SHAP de los árboles internos. Si SHAP no está
disponible, se usa un fallback transparente basado en importancia del modelo, marcado
explícitamente en `explanation_method`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from api.schemas import BusinessValueAssumptions, ExplainFeature, PredictionResponse
from api.settings import PROJECT_ROOT, settings

sys.path.append(str(PROJECT_ROOT))

log = logging.getLogger("api.predict")

DEFAULT_METADATA = {
    "model_name": "modelo_final_dp261",
    "model_version": settings.model_version,
    "threshold": 0.5,
    "business_assumptions": {"benefit_tp": 2500, "benefit_tn": 600, "cost_fp": -900, "cost_fn": -4500},
}

FEATURE_FRIENDLY_NAMES = {
    "VehOdo": "Kilometraje del vehículo",
    "VehicleAge": "Antigüedad del vehículo",
    "VehBCost": "Costo base de compra",
    "WarrantyCost": "Costo de garantía",
    "Auction": "Casa de subasta",
    "Make": "Marca",
    "Model": "Modelo",
    "Size": "Tamaño / segmento",
    "Nationality": "Procedencia / nacionalidad",
    "TopThreeAmericanName": "Grupo de marca",
    "MMRAcquisitionAuctionAveragePrice": "Precio MMR de adquisición en subasta",
    "MMRAcquisitionAuctionCleanPrice": "Precio MMR limpio de adquisición",
    "MMRCurrentAuctionAveragePrice": "Precio MMR actual en subasta",
    "MMRCurrentAuctionCleanPrice": "Precio MMR limpio actual",
    "PRIMEUNIT": "Indicador PRIMEUNIT",
    "AUCGUART": "Indicador AUCGUART",
    "VNST": "Estado de venta",
    "IsOnlineSale": "Venta online",
    "odo_per_year": "Kilometraje por año",
    "old_high_mileage_flag": "Vehículo antiguo con alto kilometraje",
    "cost_to_acq_auction_avg": "Costo vs. precio MMR de adquisición",
    "acq_auction_margin": "Margen frente a MMR de adquisición",
    "cost_to_current_auction_avg": "Costo vs. MMR actual",
    "current_auction_margin": "Margen frente a MMR actual",
    "warranty_to_cost": "Garantía como proporción del costo",
    "warranty_per_vehicle_year": "Garantía por año de antigüedad",
    "auction_avg_depreciation": "Depreciación estimada en subasta",
    "retail_avg_depreciation": "Depreciación estimada retail",
    "acq_clean_avg_spread": "Spread adquisición clean vs. average",
    "current_clean_avg_spread": "Spread actual clean vs. average",
    "mmr_missing_count": "Cantidad de precios MMR faltantes",
}


def _to_plain_python(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-safe Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


@lru_cache(maxsize=1)
def load_metadata() -> Dict[str, Any]:
    if settings.metadata_path.exists():
        return json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    return DEFAULT_METADATA


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not settings.model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {settings.model_path}. Ejecuta `python scripts/run_all.py`.")
    return joblib.load(settings.model_path)


def model_sha() -> str:
    if not settings.model_path.exists():
        return "model-not-found"
    h = hashlib.sha256()
    with open(settings.model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_plain_python(v) for k, v in payload.items() if k != "IsBadBuy"}


def payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([sanitize_payload(payload)])


def score_model(model: Any, X_raw: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_raw)
        return np.asarray(proba)[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X_raw), dtype=float)
        return 1 / (1 + np.exp(-raw))
    return np.asarray(model.predict(X_raw), dtype=float)


def risk_segment(score: float, threshold: float) -> Tuple[str, str]:
    """Commercial decision bands.

    ROJO uses the final model threshold. ÁMBAR is intentionally conservative: cases close
    to the threshold should be manually reviewed instead of accepted automatically.
    """
    if score >= threshold:
        return "ROJO", "ROJO - Alto riesgo: detener compra o derivar a revisión experta"
    if score >= max(threshold - 0.20, 0.0):
        return "ÁMBAR", "ÁMBAR - Riesgo medio: revisar manualmente antes de comprar"
    return "VERDE", "VERDE - Riesgo bajo: puede continuar evaluación comercial"


def _preprocess(model: Any):
    return model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None


def _features_step(model: Any):
    return model.named_steps.get("features") if hasattr(model, "named_steps") else None


def _classifier(model: Any):
    return model.named_steps.get("clf") if hasattr(model, "named_steps") else model


def _feature_names(model: Any, X_raw: pd.DataFrame) -> List[str]:
    preprocess = _preprocess(model)
    if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
        try:
            return [str(x) for x in preprocess.get_feature_names_out()]
        except Exception:
            pass
    return [str(x) for x in X_raw.columns]


def _feature_frame(model: Any, X_raw: pd.DataFrame) -> pd.DataFrame:
    features = _features_step(model)
    if features is None:
        return X_raw.copy()
    X_features = features.transform(X_raw)
    if isinstance(X_features, pd.DataFrame):
        return X_features
    return pd.DataFrame(X_features)


def _transformed_row(model: Any, X_raw: pd.DataFrame):
    preprocess = _preprocess(model)
    if preprocess is None:
        return None
    X_features = _feature_frame(model, X_raw)
    return preprocess.transform(X_features)


def _dense_vector(row: Any) -> np.ndarray:
    return row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()


def _extract_class1_shap_values(values: Any) -> np.ndarray:
    """Normaliza la salida de SHAP y extrae contribuciones de la clase positiva.

    SHAP puede devolver distintos formatos según versión/modelo:
    - list[class_0, class_1]
    - array con shape (n_samples, n_features)
    - array con shape (n_samples, n_features, n_classes)
    - Explanation.values
    """
    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        values = values[-1]

    arr = np.asarray(values)

    if arr.ndim == 3:
        # Caso frecuente en SHAP reciente: (n_samples, n_features, n_outputs)
        if arr.shape[0] == 1 and arr.shape[-1] >= 2:
            return arr[0, :, 1].astype(float).ravel()
        # Alternativa: (n_outputs, n_samples, n_features)
        if arr.shape[0] >= 2 and arr.shape[1] == 1:
            return arr[-1, 0, :].astype(float).ravel()

    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0].astype(float).ravel()
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float).ravel()

    return arr.astype(float).ravel()


def _slice_transformed_row(row: Any, selected_features: np.ndarray) -> Any:
    """Obtiene las columnas usadas por un estimador interno de Bagging."""
    try:
        return row[:, selected_features]
    except Exception:
        dense = _dense_vector(row)
        return dense[selected_features].reshape(1, -1)


def _bagging_tree_shap_impacts(clf: Any, row: Any, n_features: int) -> np.ndarray | None:
    """Promedia explicaciones SHAP locales de los árboles internos de Bagging.

    BaggingClassifier no siempre es soportado por TreeExplainer como meta-estimador.
    En lugar de caer directamente a importancia global, se explica cada árbol interno
    y se promedian sus contribuciones locales. Esto mantiene el foco correcto:
    cuánto empuja cada variable la predicción de ESTE vehículo.
    """
    if not hasattr(clf, "estimators_"):
        return None

    try:
        import shap  # type: ignore
    except Exception:
        return None

    estimators_features = getattr(clf, "estimators_features_", None)
    total_impacts = np.zeros(n_features, dtype=float)
    valid_estimators = 0

    for idx, estimator in enumerate(clf.estimators_):
        selected = (
            np.asarray(estimators_features[idx], dtype=int)
            if estimators_features is not None
            else np.arange(n_features, dtype=int)
        )

        try:
            row_subset = _slice_transformed_row(row, selected)
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(row_subset)
            local_values = _extract_class1_shap_values(values)

            if len(local_values) == 0:
                continue

            # Alinear la explicación del árbol interno al espacio transformado completo.
            usable = min(len(local_values), len(selected))
            mapped = np.zeros(n_features, dtype=float)
            mapped[selected[:usable]] = local_values[:usable]

            total_impacts += mapped
            valid_estimators += 1
        except Exception as exc:
            log.debug("SHAP failed for Bagging inner estimator %s: %s", idx, exc)
            continue

    if valid_estimators == 0:
        return None

    return total_impacts / valid_estimators


def _bagging_feature_importances(clf: Any, n_features: int) -> np.ndarray | None:
    """Average feature importances across the inner estimators of a BaggingClassifier."""
    if not hasattr(clf, "estimators_"):
        return None
    importances = np.zeros(n_features, dtype=float)
    valid_estimators = 0
    estimators_features = getattr(clf, "estimators_features_", None)

    for idx, estimator in enumerate(clf.estimators_):
        if not hasattr(estimator, "feature_importances_"):
            continue
        fi = np.asarray(estimator.feature_importances_, dtype=float).ravel()
        if estimators_features is not None:
            selected_features = np.asarray(estimators_features[idx], dtype=int)
            mapped = np.zeros(n_features, dtype=float)
            mapped[selected_features[: len(fi)]] = fi
            fi = mapped
        elif len(fi) != n_features:
            # Avoid returning misaligned explanations.
            continue
        importances += fi
        valid_estimators += 1

    if valid_estimators == 0:
        return None
    importances = importances / valid_estimators
    total = importances.sum()
    return importances / total if total > 0 else importances


def _classifier_feature_importances(clf: Any, n_features: int) -> Tuple[np.ndarray | None, str]:
    if hasattr(clf, "feature_importances_"):
        fi = np.asarray(clf.feature_importances_, dtype=float).ravel()
        if len(fi) == n_features:
            total = fi.sum()
            return (fi / total if total > 0 else fi), "feature_importance_fallback"

    bagging_fi = _bagging_feature_importances(clf, n_features)
    if bagging_fi is not None:
        return bagging_fi, "bagging_tree_importance_fallback"

    if hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_).ravel()
        if len(coefs) == n_features:
            return coefs, "linear_coefficient_fallback"

    return None, "unavailable: classifier has no supported local explanation attribute"


def _base_feature_name(transformed_feature: str) -> str:
    """Convert ColumnTransformer/OHE names into a readable base feature."""
    name = transformed_feature
    if "__" in name:
        name = name.split("__", 1)[1]
    # OneHotEncoder names often look like Make_FORD or Auction_MANHEIM.
    # Keep engineered features intact when they already exist in the dictionary.
    if name in FEATURE_FRIENDLY_NAMES:
        return name
    for prefix in sorted(FEATURE_FRIENDLY_NAMES, key=len, reverse=True):
        if name.startswith(prefix + "_"):
            return prefix
    return name


def _friendly_feature_label(transformed_feature: str) -> str:
    base = _base_feature_name(transformed_feature)
    label = FEATURE_FRIENDLY_NAMES.get(base, base)
    if base != transformed_feature and transformed_feature.endswith(tuple(["_Y", "_N"])):
        return label
    return label


def _raw_feature_value(base_feature: str, X_features: pd.DataFrame, X_raw: pd.DataFrame, dense_value: float) -> Any:
    if base_feature in X_features.columns:
        return _to_plain_python(X_features.iloc[0][base_feature])
    if base_feature in X_raw.columns:
        return _to_plain_python(X_raw.iloc[0][base_feature])
    return float(dense_value) if np.isfinite(dense_value) else None


def _compress_explanations(
    names: Iterable[str],
    impacts: np.ndarray,
    dense_row: np.ndarray,
    X_features: pd.DataFrame,
    X_raw: pd.DataFrame,
    top_n: int,
    method: str,
) -> List[ExplainFeature]:
    """Aggregate transformed features back to human-readable business variables."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, transformed_name in enumerate(names):
        if idx >= len(impacts) or idx >= len(dense_row):
            continue
        impact = float(impacts[idx])
        if not np.isfinite(impact):
            continue
        base = _base_feature_name(transformed_name)
        label = FEATURE_FRIENDLY_NAMES.get(base, base)
        current = grouped.setdefault(
            base,
            {
                "feature": base,
                "display_name": label,
                "value": _raw_feature_value(base, X_features, X_raw, dense_row[idx]),
                "impact": 0.0,
                "abs_impact": 0.0,
                "detail": transformed_name,
            },
        )
        current["impact"] += impact
        current["abs_impact"] += abs(impact)
        # Keep the transformed detail with the highest absolute contribution.
        if abs(impact) > abs(float(current.get("detail_impact", 0.0))):
            current["detail"] = transformed_name
            current["detail_impact"] = impact

    ordered = sorted(grouped.values(), key=lambda x: x["abs_impact"], reverse=True)[:top_n]
    items: List[ExplainFeature] = []
    for item in ordered:
        impact = float(item["impact"])
        direction = "sube_riesgo" if impact >= 0 else "baja_riesgo"
        # For pure feature-importance fallbacks, direction is approximate, because tree importances are unsigned.
        if method in {"feature_importance_fallback", "bagging_tree_importance_fallback"}:
            direction = "factor_relevante"
        items.append(
            ExplainFeature(
                feature=str(item["feature"]),
                display_name=str(item["display_name"]),
                value=item.get("value"),
                impact=impact,
                impact_abs=float(item["abs_impact"]),
                impact_direction=direction,
                detail=str(item.get("detail", "")),
            )
        )
    return items


def explain_prediction(model: Any, X_raw: pd.DataFrame, top_n: int = 5) -> Tuple[List[ExplainFeature], str]:
    """Devuelve factores locales que explican la predicción de un vehículo.

    Prioridad:
    1. Bagging sobre árboles: promedio de SHAP local de los árboles internos.
    2. Otros árboles/boosting compatibles: SHAP TreeExplainer directo.
    3. Fallback: importancia global ponderada por valores activos de la instancia.
    """
    try:
        clf = _classifier(model)
        row = _transformed_row(model, X_raw)
        if row is None:
            return [], "unavailable: preprocessing step not found"

        names = _feature_names(model, X_raw)
        dense_row = _dense_vector(row)
        X_features = _feature_frame(model, X_raw)

        impacts: np.ndarray | None = None
        method = "unavailable"

        # 1) Caso especial: BaggingClassifier no siempre es soportado como meta-estimador.
        bagging_shap = _bagging_tree_shap_impacts(clf, row, len(dense_row))
        if bagging_shap is not None:
            impacts = bagging_shap
            method = "shap.TreeExplainer_bagging_average"

        # 2) SHAP directo para árboles/boosting compatibles.
        if impacts is None:
            try:
                import shap  # type: ignore

                explainer = shap.TreeExplainer(clf)
                values = explainer.shap_values(row)
                impacts = _extract_class1_shap_values(values)
                method = "shap.TreeExplainer"
            except Exception as exc:
                log.warning("SHAP failed; using fallback explanation: %s", exc)

        # 3) Fallback transparente cuando SHAP no está disponible.
        if impacts is None:
            fi, method = _classifier_feature_importances(clf, len(dense_row))
            if fi is None:
                return [], method

            if method == "linear_coefficient_fallback":
                impacts = dense_row * fi
            else:
                # Importancias de árbol son no firmadas: sirven para ranking local aproximado,
                # no para dirección positiva/negativa.
                impacts = np.abs(dense_row) * fi

        impacts = np.asarray(impacts, dtype=float).ravel()
        if len(impacts) != len(dense_row):
            return [], f"unavailable: explanation length mismatch ({len(impacts)} vs {len(dense_row)})"

        return _compress_explanations(names, impacts, dense_row, X_features, X_raw, top_n, method), method
    except Exception as exc:
        return [], f"unavailable: {exc}"


def predict_one(payload: Dict[str, Any]) -> PredictionResponse:
    metadata = load_metadata()
    threshold = float(metadata.get("threshold", 0.5))
    model_name = str(metadata.get("model_name", "modelo_final_dp261"))
    model_version = str(metadata.get("model_version", settings.model_version))
    assumptions = metadata.get("business_assumptions", DEFAULT_METADATA["business_assumptions"])

    model = load_model()
    X_raw = payload_to_df(payload)
    risk_score = float(score_model(model, X_raw)[0])
    prediction = int(risk_score >= threshold)
    segment, decision = risk_segment(risk_score, threshold)
    top_features, method = explain_prediction(model, X_raw)

    return PredictionResponse(
        risk_score=risk_score,
        threshold=threshold,
        prediction=prediction,
        risk_segment=segment,
        decision=decision,
        model_name=model_name,
        model_version=model_version,
        api_version=settings.api_version,
        business_value_assumptions=BusinessValueAssumptions(**assumptions),
        shap_top_features=top_features,
        explanation_method=method,
    )
