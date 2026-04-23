from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

import joblib
import pandas as pd


def load_model_artifact(path: str | Path) -> Any:
    """Carga un artefacto serializado con joblib."""
    return joblib.load(Path(path))


def save_model_artifact(obj: Any, path: str | Path) -> None:
    """Guarda un artefacto serializado con joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)


def predict_with_model(model: Any, X):
    """Genera predicciones con un modelo cargado."""
    return model.predict(X)


def validate_model_artifact(path: str | Path, X_sample) -> dict:
    """Valida que un artefacto pueda cargarse y predecir."""
    artifact_path = Path(path)
    model = load_model_artifact(artifact_path)
    preds = predict_with_model(model, X_sample)
    return {
        "artifact_path": str(artifact_path),
        "exists": artifact_path.exists(),
        "n_sample_predictions": len(preds),
        "sample_predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds),
    }


def load_experiment_log(path: str | Path) -> pd.DataFrame:
    """Carga el log de experimentos si existe; si no, devuelve un DataFrame vacío."""
    log_path = Path(path)
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame()


def append_experiment_log(row: dict, path: str | Path) -> pd.DataFrame:
    """Agrega un registro al log de experimentos y devuelve el DataFrame actualizado."""
    log_path = Path(path)
    existing = load_experiment_log(log_path)

    row = dict(row)
    row.setdefault("log_timestamp", datetime.now().isoformat(timespec="seconds"))

    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(log_path, index=False)
    return updated
