"""Runtime settings for the Sprint 6 FastAPI service.

All values can be overridden with environment variables so secrets and paths are
not hardcoded in the repository.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_version: str = os.getenv("API_VERSION", "1.0.0")
    model_version: str = os.getenv("MODEL_VERSION", "sprint5-final-threshold-0.70")
    model_path: Path = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "final_model.pkl")))
    metadata_path: Path = Path(
        os.getenv("MODEL_METADATA_PATH", str(PROJECT_ROOT / "handoff" / "model" / "model_metadata.json"))
    )
    threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.70"))
    require_api_key: bool = os.getenv("REQUIRE_API_KEY", "false").lower() in {"1", "true", "yes"}
    api_key: str | None = os.getenv("API_KEY")
    shap_background_path: Path = Path(
        os.getenv("SHAP_BACKGROUND_PATH", str(PROJECT_ROOT / "data" / "processed" / "test_final.csv"))
    )
    shap_max_background_rows: int = int(os.getenv("SHAP_MAX_BACKGROUND_ROWS", "300"))


settings = Settings()
