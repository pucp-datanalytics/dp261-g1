"""Métricas técnicas y evaluación económica del proyecto."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BENEFIT_TN, BENEFIT_TP, BUSINESS_THRESHOLD, COST_FN, COST_FP


def get_scores(model, X):
    """Retorna score de riesgo de clase 1 compatible con varios clasificadores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1 / (1 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def business_value_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    benefit_tp: float = BENEFIT_TP,
    benefit_tn: float = BENEFIT_TN,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> float:
    """Valor total = TP*beneficio + TN*beneficio - costos FP/FN.

    Los costos ya se parametrizan con signo negativo para que la ecuación sea
    legible en las tablas y reportes.
    """
    return (tp * benefit_tp) + (tn * benefit_tn) + (fp * cost_fp) + (fn * cost_fn)


def evaluate_predictions(
    y_true,
    y_score,
    threshold: float = BUSINESS_THRESHOLD,
    benefit_tp: float = BENEFIT_TP,
    benefit_tn: float = BENEFIT_TN,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    value = business_value_from_counts(tn, fp, fn, tp, benefit_tp, benefit_tn, cost_fp, cost_fn)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "average_precision": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_value": float(value),
        "business_value_per_case": float(value) / max(len(y_true), 1),
        "positive_rate": float(y_pred.mean()),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def evaluate_model(model, X, y, threshold: float = BUSINESS_THRESHOLD) -> dict:
    return evaluate_predictions(y, get_scores(model, X), threshold=threshold)


def evaluate_threshold_grid(y_true, y_score, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, y_score, threshold=float(threshold))
        metrics.pop("classification_report", None)
        rows.append(metrics)
    return pd.DataFrame(rows)


def gain_curve(y_true, y_score):
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    return pd.DataFrame({
        "pct_population": np.arange(1, len(y_sorted) + 1) / len(y_sorted),
        "pct_positives_captured": cum_pos / max(total_pos, 1),
    })


def bootstrap_metric_ci(y_true, y_score, metric_fn, n_boot=300, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    return np.percentile(values, [2.5, 97.5]).tolist() if values else [np.nan, np.nan]
