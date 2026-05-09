from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune_candidate(
    pipe: Any,
    param_space: Dict[str, Any],
    X,
    y,
    cv,
    scoring: str = "precision",
    search_type: str = "grid",
    n_iter: int = 20,
    random_state: int = 42,
):
    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=random_state,
            return_train_score=False,
        )
    else:
        raise ValueError(f"search_type no soportado: {search_type}")

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start
    return search, elapsed


def persist_model(model: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def tuning_result_row(
    model_key: str,
    search,
    elapsed: float,
    artifact_path: str | Path,
    data_variant: str,
    search_type: str,
    primary_metric: str,
    baseline_artifact_path: str | Path | None = None,
) -> dict:
    return {
        "model_key": model_key,
        "candidate_type": "tuned",
        "data_variant": data_variant,
        "search_type": search_type,
        "primary_metric": primary_metric,
        "best_cv_score": search.best_score_,
        "best_params": json.dumps(search.best_params_),
        "fit_time_seconds": round(elapsed, 1),
        "artifact_path": str(artifact_path),
        "baseline_artifact_path": str(baseline_artifact_path) if baseline_artifact_path is not None else None,
    }


def compare_baseline_vs_tuned(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    baseline_metric_col: str = "test_precision_mean",
    tuned_metric_col: str = "best_cv_score",
) -> pd.DataFrame:
    merged = baseline_df.merge(
        tuned_df[["model_key", tuned_metric_col]],
        on="model_key",
        how="inner",
    )
    merged["improvement_abs"] = merged[tuned_metric_col] - merged[baseline_metric_col]
    merged["improvement_pct"] = np.where(
        merged[baseline_metric_col] != 0,
        merged["improvement_abs"] / merged[baseline_metric_col] * 100,
        np.nan,
    )
    return merged


def append_experiment_log(row: dict, path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    return updated
