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
from sklearn.base import clone

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



from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

def train_and_evaluate_baselines(models, X, y, model_configs, save_models=False):
    results = []
    fitted = {}
    
    for name, original_pipe in models.items():
        start = time.time()
        try:
            # 1. Filtro de columnas
            config = model_configs.get(name, {'cols': list(X.columns)})
            target_cols = config['cols']
            X_filtered = X[target_cols].copy()

            # 2. Reconstrucción total del Pipeline
            from src.preprocessing import build_preprocessor
            mode = 'linear' if name in ['LogisticRegression', 'LinearSVM', 'KNN'] else 'tree_ohe'
            
            # Forzamos un preprocesador nuevo que SOLO vea las columnas actuales
            fresh_preprocessor = build_preprocessor(X_filtered, mode=mode)
            # Clonamos solo el algoritmo (LR, RF, etc)
            fresh_model = clone(original_pipe.named_steps['model'])
            
            new_pipe = Pipeline([
                ('preprocessor', fresh_preprocessor),
                ('model', fresh_model)
            ])

            print(f">>> Evaluando {name} ({len(target_cols)} cols)...", end=" ")
            
            # 3. Evaluación rápida (n_jobs=-1 usa toda tu PC)
            # Nota: Asegúrate de que evaluate_cv use n_jobs=-1 internamente
            scores = cross_validate(
                new_pipe, X_filtered, y, 
                cv=5, 
                scoring=['recall', 'f1', 'roc_auc', 'precision'], 
                return_train_score=True, 
                n_jobs=-1 
            )
            
            # Resumen de resultados
            row = {
                "model": name,
                "recall_cv_mean": np.mean(scores['test_recall']),
                "recall_gap": np.mean(scores['train_recall']) - np.mean(scores['test_recall']),
                "f1_cv_mean": np.mean(scores['test_f1']),
                "roc_auc_cv_mean": np.mean(scores['test_roc_auc']),
                "status": "ok",
                "n_features": len(target_cols)
            }
            results.append(row)
            print("¡Listo!")
            
            if save_models:
                new_pipe.fit(X_filtered, y)
                fitted[name] = new_pipe
                
        except Exception as e:
            print(f"FALLÓ: {e}")
            results.append({"model": name, "status": "error", "error": str(e)})
            
    return pd.DataFrame(results), fitted
   

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path