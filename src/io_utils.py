"""Utilidades de entrada/salida para notebooks y scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    return path
