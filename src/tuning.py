import time

import joblib
import numpy as np
from scipy.stats import randint, loguniform, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, RANDOM_STATE
from src.models import get_cv, get_scoring
from src.preprocessing import build_preprocessor
from sklearn.svm import LinearSVC

def recall_refit(cv_results):
    recall = np.nan_to_num(cv_results["mean_test_recall"], nan=-np.inf)
    f2 = np.nan_to_num(cv_results["mean_test_f2"], nan=-np.inf)
    precision = np.nan_to_num(cv_results["mean_test_precision"], nan=-np.inf)
    ranking = np.lexsort((-precision, -f2, -recall))
    return int(ranking[0])


def build_candidate_pipeline(model_name, X):
    model_name = model_name.lower()

    if "hist" in model_name:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return Pipeline([
            ("preprocess", build_preprocessor(X, mode="ordinal")), # Boosting prefiere ordinal
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])

    if "logistic" in model_name:
        return Pipeline([
            ("preprocess", build_preprocessor(X, mode="linear")),
            ("clf", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
        ])

    if "linearsvm" in model_name or "linear_svm" in model_name or "svm" in model_name:
        return Pipeline([
            ("preprocess", build_preprocessor(X, mode="linear")),
            ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
        ])

    if "decisiontree" in model_name or "decision_tree" in model_name or "tree" in model_name:
        return Pipeline([
            ("preprocess", build_preprocessor(X, mode="tree_ohe")),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ])

    if "randomforest" in model_name or "random_forest" in model_name:
        return Pipeline([
            ("preprocess", build_preprocessor(X, mode="tree_ohe")),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ])

    if "gradient" in model_name:
        return Pipeline([
            ("preprocess", build_preprocessor(X, mode="ordinal")),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])

    raise ValueError(f"No hay pipeline definido para {model_name}")

def get_param_distributions(model_name):
    model_name = model_name.lower()

    if "logistic" in model_name:
        return {
            "clf__C": loguniform(0.01, 20),
            "clf__class_weight": [None, "balanced"],
        }

    if "linearsvm" in model_name or "linear_svm" in model_name or "svm" in model_name:
        return {
            "clf__C": loguniform(0.01, 20),
            "clf__class_weight": [None, "balanced"],
        }

    if "decisiontree" in model_name or "decision_tree" in model_name or "tree" in model_name:
        return {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 3, 5, 8, 12, 16],
            "clf__min_samples_leaf": randint(1, 30),
            "clf__min_samples_split": randint(2, 40),
            "clf__class_weight": [None, "balanced"],
        }

    if "randomforest" in model_name or "random_forest" in model_name:
        return {
            "clf__n_estimators": randint(80, 250),
            "clf__max_depth": [None, 5, 8, 12, 16],
            "clf__min_samples_leaf": randint(1, 20),
            "clf__min_samples_split": randint(2, 30),
            "clf__class_weight": [None, "balanced_subsample"],
        }

    if "gradient" in model_name:
        return {
            "clf__n_estimators": randint(80, 250),
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__max_depth": randint(2, 5),
            "clf__min_samples_leaf": randint(5, 40),
            "clf__subsample": uniform(0.65, 0.35),
        }

    return {}


def randomized_tune(model_name, X, y, n_iter=20, n_jobs=1, cv=None):
    pipe = build_candidate_pipeline(model_name, X)
    cv = cv or get_cv()

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=get_param_distributions(model_name),
        n_iter=n_iter,
        scoring=get_scoring(),
        refit=recall_refit,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
        error_score=np.nan,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"tuned_{model_name}.pkl"
    joblib.dump(search.best_estimator_, path)

    best_idx = search.best_index_
    cv_results = search.cv_results_

    return {
        "model": model_name,
        "best_index": int(best_idx),
        "best_params": search.best_params_,
        "best_recall": cv_results["mean_test_recall"][best_idx],
        "best_f2": cv_results["mean_test_f2"][best_idx],
        "best_precision": cv_results["mean_test_precision"][best_idx],
        "best_f1": cv_results["mean_test_f1"][best_idx],
        "best_f05": cv_results["mean_test_f05"][best_idx],
        "best_roc_auc": cv_results["mean_test_roc_auc"][best_idx],
        "best_average_precision": cv_results["mean_test_average_precision"][best_idx],
        "train_recall": cv_results["mean_train_recall"][best_idx],
        "train_f2": cv_results["mean_train_f2"][best_idx],
        "recall_gap_train_minus_cv": cv_results["mean_train_recall"][best_idx] - cv_results["mean_test_recall"][best_idx],
        "f2_gap_train_minus_cv": cv_results["mean_train_f2"][best_idx] - cv_results["mean_test_f2"][best_idx],
        "seconds": round(elapsed, 1),
        "path": str(path),
    }, search


def optuna_tune_gradient_boosting(X, y, n_trials=20):
    import optuna

    cv = get_cv()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        }
        pipe = Pipeline([
            ("preprocess", build_preprocessor(X, mode="ordinal")),
            ("clf", GradientBoostingClassifier(**params, random_state=RANDOM_STATE)),
        ])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=get_scoring(), n_jobs=1)
        recall = scores["test_recall"].mean()
        f2 = scores["test_f2"].mean()
        precision = scores["test_precision"].mean()
        return recall + 1e-3 * f2 + 1e-6 * precision

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_pipe = Pipeline([
        ("preprocess", build_preprocessor(X, mode="ordinal")),
        ("clf", GradientBoostingClassifier(**study.best_params, random_state=RANDOM_STATE)),
    ])
    best_pipe.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "tuned_gradient_boosting_optuna.pkl"
    joblib.dump(best_pipe, path)

    return study, best_pipe, path


def build_simple_ensembles(fitted_estimators, X=None):
    estimators = [(name, model) for name, model in fitted_estimators.items()]
    if len(estimators) < 2:
        return {}

    return {
        "Voting_hard": VotingClassifier(estimators=estimators, voting="hard"),
        "Stacking_lr": StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=3,
        ),
    }
