import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from src.config import BENEFIT_TN, BENEFIT_TP, COST_FN, COST_FP


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return model.predict(X)


def build_threshold_grid(y_score, score_type="predict_proba"):
    y_score = np.asarray(y_score)
    if score_type in {"predict_proba", "decision_function_scaled"}:
        return np.round(np.arange(0.01, 0.91, 0.01), 3)

    finite_scores = y_score[np.isfinite(y_score)]
    return np.unique(np.quantile(finite_scores, np.linspace(0.01, 0.99, 99)))


def get_oof_scores(estimator, X, y, cv, n_jobs=1):
    if hasattr(estimator, "predict_proba"):
        scores = cross_val_predict(
            estimator,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1]
        return np.asarray(scores), "predict_proba"

    if hasattr(estimator, "decision_function"):
        raw = cross_val_predict(
            estimator,
            X,
            y,
            cv=cv,
            method="decision_function",
            n_jobs=n_jobs,
        )
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return np.asarray(scores), "decision_function_scaled"

    scores = cross_val_predict(estimator, X, y, cv=cv, method="predict", n_jobs=n_jobs)
    return np.asarray(scores), "predict"


def evaluate_thresholds(
    y_true,
    y_score,
    thresholds=None,
    benefit_tp=BENEFIT_TP,
    cost_fp=COST_FP,
    cost_fn=COST_FN,
    benefit_tn=BENEFIT_TN,
):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        business_value = (
            tp * benefit_tp +
            fp * cost_fp +
            fn * cost_fn +
            tn * benefit_tn
        )

        rows.append({
            "threshold": thr,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f05": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
            "fn_rate": fn / max(fn + tp, 1),
            "fp_rate": fp / max(fp + tn, 1),
            "business_value": business_value,
            "positive_rate": y_pred.mean(),
        })

    return pd.DataFrame(rows)


def select_best_threshold(threshold_results, min_precision=None, max_positive_rate=None):
    candidates = threshold_results.copy()

    if min_precision is not None:
        candidates = candidates[candidates["precision"] >= min_precision]

    if max_positive_rate is not None:
        candidates = candidates[candidates["positive_rate"] <= max_positive_rate]

    if candidates.empty:
        candidates = threshold_results.copy()

    return candidates.sort_values(
        ["recall", "f2", "precision"],
        ascending=[False, False, False],
    ).head(1)


def threshold_tuning_cv(
    model_name,
    estimator,
    X,
    y,
    cv,
    thresholds=None,
    n_jobs=1,
    min_precision=None,
    max_positive_rate=None,
):
    scores, score_type = get_oof_scores(estimator, X, y, cv=cv, n_jobs=n_jobs)

    if thresholds is None:
        thresholds = build_threshold_grid(scores, score_type)

    results = evaluate_thresholds(y, scores, thresholds=thresholds)
    results.insert(0, "model", model_name)
    results["score_type"] = score_type

    best = select_best_threshold(
        results,
        min_precision=min_precision,
        max_positive_rate=max_positive_rate,
    ).copy()
    best["selected"] = True

    return results


def final_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f05": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def bootstrap_metric_ci(y_true, y_score, metric_fn, n_boot=300, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))

    return np.percentile(vals, [2.5, 97.5]).tolist()


def gain_curve(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    pct_population = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    pct_positives = cum_pos / max(total_pos, 1)

    return pd.DataFrame({
        "pct_population": pct_population,
        "pct_positives_captured": pct_positives,
    })
