"""Runtime settings for FastAPI service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_version: str = os.getenv("API_VERSION", "1.0.0")
    model_version: str = os.getenv("MODEL_VERSION", "final-entrega-dp261-v1")
    model_path: Path = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "final_model.pkl")))
    metadata_path: Path = Path(os.getenv("MODEL_METADATA_PATH", str(PROJECT_ROOT / "models" / "model_metadata.json")))
    require_api_key: bool = os.getenv("REQUIRE_API_KEY", "false").lower() in {"1", "true", "yes"}
    api_key: str | None = os.getenv("API_KEY")
    shap_max_background_rows: int = int(os.getenv("SHAP_MAX_BACKGROUND_ROWS", "200"))


settings = Settings()
