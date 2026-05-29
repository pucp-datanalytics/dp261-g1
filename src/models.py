"""Construcción, entrenamiento y persistencia de modelos.

Este módulo sigue los modelos solicitados en los sprints:
- Baseline: Logistic Regression, Decision Tree, Random Forest, SVM y KNN.
- Avanzados: XGBoost, LightGBM y ensambles Voting/Bagging/Stacking.

No se incluyen modelos fuera del alcance del backlog, como ExtraTrees, para que
la sustentación sea consistente con los PDFs del curso.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_SPLITS, MODELS_DIR, RANDOM_STATE
from src.preprocessing import FeatureEngineeringTransformer, build_preprocessor


def get_cv(n_splits: int = CV_SPLITS):
    """Cross-validation estratificada para preservar la tasa de Bad Buy por fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def get_scoring():
    """Métricas técnicas alineadas al negocio.

    Recall y F2 son relevantes porque el falso negativo es el error más caro.
    Precision se monitorea para no bloquear demasiadas buenas compras.
    """
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }


def make_model_pipeline(X_raw: pd.DataFrame, mode: str, classifier, sampler=None, min_frequency: float = 0.01):
    """Crea un pipeline completo: features -> preprocesamiento -> balanceo opcional -> modelo.

    El balanceo, cuando aplica, queda dentro del pipeline de imbalanced-learn para
    que se ejecute solo sobre train/folds de entrenamiento y no contamine validación/test.
    """
    steps = [
        ("features", FeatureEngineeringTransformer()),
        ("preprocess", build_preprocessor(X_raw, mode=mode, min_frequency=min_frequency)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))
        steps.append(("clf", classifier))
        return ImbPipeline(steps)
    steps.append(("clf", classifier))
    return Pipeline(steps)


def make_under_sampler(sampling_strategy: float = 0.55):
    """Submuestreo controlado de la clase mayoritaria.

    Se prefiere sobre SMOTE para modelos lineales/distancia porque el dataset tiene
    muchas categóricas codificadas con OHE; generar vecinos sintéticos sobre dummies
    puede crear combinaciones poco realistas de marca/modelo/subasta.
    """
    return RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)


def class_ratio(y: pd.Series) -> float:
    y = pd.Series(y).astype(int)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return negatives / positives


def build_baseline_models(X_raw: pd.DataFrame) -> dict:
    """Modelos baseline solicitados en Sprint 3.

    - Logistic Regression, SVM y KNN: requieren escalado; usan RobustScaler/StandardScaler
      desde `build_preprocessor(..., mode='linear')` y submuestreo controlado.
    - Decision Tree y Random Forest: no requieren escalado; usan pesos de clase.
    """
    sampler = make_under_sampler(0.55)
    return {
        "LogisticRegression_baseline": make_model_pipeline(
            X_raw,
            "linear",
            LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE),
            sampler=sampler,
            min_frequency=0.015,
        ),
        "DecisionTree_baseline": make_model_pipeline(
            X_raw,
            "tree_ohe",
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            min_frequency=0.015,
        ),
        "RandomForest_baseline": make_model_pipeline(
            X_raw,
            "tree_ohe",
            RandomForestClassifier(n_estimators=20, class_weight="balanced_subsample", n_jobs=1, random_state=RANDOM_STATE),
            min_frequency=0.015,
        ),
        "SVM_baseline": make_model_pipeline(
            X_raw,
            "linear",
            SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE, max_iter=3000),
            sampler=sampler,
            min_frequency=0.02,
        ),
        "KNN_baseline": make_model_pipeline(
            X_raw,
            "linear",
            KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=1),
            sampler=sampler,
            min_frequency=0.02,
        ),
    }


def build_advanced_models(X_raw: pd.DataFrame, y_train: pd.Series | None = None) -> dict:
    """Modelos avanzados solicitados en Sprint 4: XGBoost, LightGBM y Bagging.

    XGBoost/LightGBM reciben variables categóricas ordinalizadas para evitar una matriz OHE
    muy grande. La variable `scale_pos_weight` compensa el desbalance sin re-muestrear.
    """
    scale_pos_weight = class_ratio(y_train) if y_train is not None else 7.0
    models = {}

    try:
        from xgboost import XGBClassifier

        models["XGBoost_baseline"] = make_model_pipeline(
            X_raw,
            "ordinal",
            XGBClassifier(
                n_estimators=30,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM_baseline"] = make_model_pipeline(
            X_raw,
            "ordinal",
            LGBMClassifier(
                n_estimators=40,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
        )
    except Exception:
        pass

    base_tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE)
    try:
        models["Bagging_DecisionTree"] = make_model_pipeline(
            X_raw,
            "tree_ohe",
            BaggingClassifier(
                estimator=base_tree,
                n_estimators=20,
                max_samples=0.8,
                max_features=0.9,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            min_frequency=0.02,
        )
    except TypeError:  # sklearn antiguo usa base_estimator
        models["Bagging_DecisionTree"] = make_model_pipeline(
            X_raw,
            "tree_ohe",
            BaggingClassifier(
                base_estimator=base_tree,
                n_estimators=20,
                max_samples=0.8,
                max_features=0.9,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            min_frequency=0.02,
        )

    return models


def build_ensembles(X_raw: pd.DataFrame, y_train: pd.Series | None = None) -> dict:
    """Ensambles Voting y Stacking solicitados en Sprint 4.

    A diferencia de un único preprocesador global, aquí cada estimador base es
    un pipeline completo. Así, Logistic Regression recibe escalado, Random Forest
    recibe OHE sin escalado y LightGBM recibe codificación ordinal. Esto respeta
    la regla de pipelines diferenciados por familia de algoritmo.
    """
    ratio = class_ratio(y_train) if y_train is not None else 7.0
    lr_pipe = make_model_pipeline(
        X_raw,
        "linear",
        LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE),
        sampler=make_under_sampler(0.55),
        min_frequency=0.02,
    )
    rf_pipe = make_model_pipeline(
        X_raw,
        "tree_ohe",
        RandomForestClassifier(
            n_estimators=20,
            max_depth=12,
            min_samples_leaf=30,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        min_frequency=0.02,
    )
    try:
        from lightgbm import LGBMClassifier
        gbm_pipe = make_model_pipeline(
            X_raw,
            "ordinal",
            LGBMClassifier(
                n_estimators=30,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=ratio,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
        )
    except Exception:
        gbm_pipe = make_model_pipeline(
            X_raw,
            "tree_ohe",
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE),
            min_frequency=0.02,
        )

    voting = VotingClassifier(
        estimators=[("lr", lr_pipe), ("rf", rf_pipe), ("gbm", gbm_pipe)],
        voting="soft",
        n_jobs=1,
    )
    stacking = StackingClassifier(
        estimators=[("rf", rf_pipe), ("gbm", gbm_pipe)],
        final_estimator=LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
        cv=3,
        stack_method="predict_proba",
        n_jobs=1,
    )
    return {
        "Voting_soft_LR_RF_LGBM": voting,
        "Stacking_RF_LGBM_LR": stacking,
    }

def evaluate_cv(pipe, X, y, cv=None, scoring=None):
    cv = cv or get_cv()
    scoring = scoring or get_scoring()
    return cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=1)


def summarize_cv_scores(name: str, scores: dict) -> dict:
    row = {"model": name, "fit_time_mean": float(np.mean(scores["fit_time"]))}
    metric_names = [key.replace("test_", "") for key in scores if key.startswith("test_")]
    for metric in metric_names:
        row[f"{metric}_cv_mean"] = float(np.mean(scores[f"test_{metric}"]))
        row[f"{metric}_cv_std"] = float(np.std(scores[f"test_{metric}"]))
        train_key = f"train_{metric}"
        if train_key in scores:
            row[f"{metric}_train_mean"] = float(np.mean(scores[train_key]))
            row[f"{metric}_gap"] = row[f"{metric}_train_mean"] - row[f"{metric}_cv_mean"]
    return row


def train_and_score_candidates(models: dict, X_train, y_train, X_valid, y_valid, evaluator: Callable):
    rows = []
    fitted = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        metrics = evaluator(model, X_valid, y_valid)
        metrics.pop("classification_report", None)
        row = {"model": name, "seconds": round(time.time() - start, 1), **metrics}
        rows.append(row)
        fitted[name] = model
    return pd.DataFrame(rows), fitted


def save_model(model, path: Path | str = MODELS_DIR / "final_model.pkl"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
