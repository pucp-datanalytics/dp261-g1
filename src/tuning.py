"""Utilidades de optimización de hiperparámetros.

Se implementan dos enfoques pedidos en Sprint 4:
1. RandomizedSearchCV: exploración rápida de espacios grandes.
2. Optuna / optimización bayesiana TPE: refinamiento inteligente de LightGBM.

Todos los objetos tuneados son pipelines completos, por lo que el preprocesamiento
se ajusta dentro de cada fold y se evita data leakage.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.config import OPTUNA_TRIALS, RANDOM_SEARCH_ITER, RANDOM_STATE
from src.models import class_ratio, get_cv, make_model_pipeline

F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)


def tune_with_random_search(name: str, pipe, param_distributions: dict, X, y, n_iter: int = RANDOM_SEARCH_ITER) -> tuple[Any, dict]:
    """Ejecuta RandomizedSearchCV con métrica F2 para priorizar recall de clase 1."""
    start = time.time()
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=F2_SCORER,
        cv=get_cv(),
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=0,
        error_score="raise",
    )
    search.fit(X, y)
    row = {
        "model": name,
        "tuning_method": "RandomizedSearchCV",
        "best_cv_f2": float(search.best_score_),
        "best_params": search.best_params_,
        "seconds": round(time.time() - start, 1),
    }
    return search.best_estimator_, row


def tune_lightgbm_with_optuna(X_raw, X, y, n_trials: int = OPTUNA_TRIALS) -> tuple[Any, dict]:
    """Optimización bayesiana TPE con Optuna para LightGBM.

    Se tunean hiperparámetros que controlan complejidad, tasa de aprendizaje,
    regularización y muestreo. El objetivo maximiza F2 en CV porque el costo FN
    es mayor que el costo FP.
    """
    import optuna
    from lightgbm import LGBMClassifier

    ratio = class_ratio(y)
    cv = get_cv()
    start = time.time()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 80),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 12, 48),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.0, log=True),
            "scale_pos_weight": ratio,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbose": -1,
        }
        clf = LGBMClassifier(**params)
        pipe = make_model_pipeline(X_raw, "ordinal", clf)
        score = cross_val_score(pipe, X, y, scoring=F2_SCORER, cv=cv, n_jobs=1).mean()
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    best_params.update({"scale_pos_weight": ratio, "random_state": RANDOM_STATE, "n_jobs": 1, "verbose": -1})
    best_model = make_model_pipeline(X_raw, "ordinal", LGBMClassifier(**best_params))
    best_model.fit(X, y)
    row = {
        "model": "LightGBM_Optuna_TPE",
        "tuning_method": "Optuna_TPE_Bayesian",
        "best_cv_f2": float(study.best_value),
        "best_params": best_params,
        "seconds": round(time.time() - start, 1),
        "n_trials": n_trials,
    }
    trials_df = study.trials_dataframe()
    return best_model, row, trials_df


def default_random_search_spaces() -> dict:
    """Espacios de búsqueda compactos para ejecutar en laptop."""
    return {
        "RandomForest_random_search": {
            "clf__n_estimators": randint(40, 100),
            "clf__max_depth": randint(5, 18),
            "clf__min_samples_leaf": randint(10, 80),
            "clf__min_samples_split": randint(20, 140),
            "clf__max_features": ["sqrt", "log2", 0.7],
        },
        "DecisionTree_random_search": {
            "clf__max_depth": randint(4, 16),
            "clf__min_samples_leaf": randint(20, 160),
            "clf__min_samples_split": randint(30, 200),
            "clf__criterion": ["gini", "entropy", "log_loss"],
        },
        "LogisticRegression_random_search": {
            "clf__C": loguniform(0.02, 8.0),
            "clf__penalty": ["l1", "l2"],
            "sampler__sampling_strategy": uniform(0.35, 0.35),
        },
        "XGBoost_random_search": {
            "clf__n_estimators": randint(30, 90),
            "clf__max_depth": randint(3, 7),
            "clf__learning_rate": loguniform(0.02, 0.15),
            "clf__subsample": uniform(0.65, 0.30),
            "clf__colsample_bytree": uniform(0.65, 0.30),
        },
    }


def dataframe_from_rows(rows: list[dict]) -> pd.DataFrame:
    out = []
    for row in rows:
        clean = row.copy()
        clean["best_params"] = str(clean.get("best_params", {}))
        out.append(clean)
    return pd.DataFrame(out)
