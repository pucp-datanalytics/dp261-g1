import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_SPLITS, MODELS_DIR, RANDOM_STATE
from src.preprocessing import build_preprocessor


def get_scoring():
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "f05": make_scorer(fbeta_score, beta=0.5, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

def get_cv(n_splits=CV_SPLITS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def build_baseline_models(X_sample):
    linear_prep = build_preprocessor(X_sample, mode="linear")
    tree_prep = build_preprocessor(X_sample, mode="tree_ohe")
    ordinal_prep = build_preprocessor(X_sample, mode="ordinal")

    return {
        "Dummy_most_frequent": Pipeline([
            ("preprocess", linear_prep),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "LogisticRegression": Pipeline([
            ("preprocess", linear_prep),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "DecisionTree": Pipeline([
            ("preprocess", tree_prep),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("preprocess", tree_prep),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "LinearSVM": Pipeline([
            ("preprocess", linear_prep),
            ("clf", LinearSVC(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("preprocess", linear_prep),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "GradientBoosting": Pipeline([
            ("preprocess", ordinal_prep),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocess", ordinal_prep),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_cv(pipe, X, y, cv=None, scoring=None):
    cv = cv or get_cv()
    scoring = scoring or get_scoring()
    return cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=1)


def summarize_cv_scores(name, scores):
    row = {"model": name, "fit_time_mean": scores["fit_time"].mean()}
    metric_names = [k.replace("test_", "") for k in scores if k.startswith("test_")]
    for metric in metric_names:
        row[f"{metric}_cv_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_cv_std"] = scores[f"test_{metric}"].std()
        train_key = f"train_{metric}"
        if train_key in scores:
            row[f"{metric}_train_mean"] = scores[train_key].mean()
            row[f"{metric}_gap"] = row[f"{metric}_train_mean"] - row[f"{metric}_cv_mean"]
    return row


def train_and_evaluate_baselines(models, X, y, save_models=False):
    results = []
    fitted = {}
    for name, pipe in models.items():
        start = time.time()
        try:
            scores = evaluate_cv(pipe, X, y)
            row = summarize_cv_scores(name, scores)
            row["status"] = "ok"
            row["seconds_total"] = round(time.time() - start, 1)
            results.append(row)
            if save_models:
                pipe.fit(X, y)
                path = MODELS_DIR / f"baseline_{name}.pkl"
                joblib.dump(pipe, path)
                row["model_path"] = str(path)
                fitted[name] = pipe
        except Exception as exc:
            results.append({"model": name, "status": "error", "error": str(exc), "seconds_total": round(time.time() - start, 1)})
    return pd.DataFrame(results), fitted


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
