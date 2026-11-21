"""Runtime configuration (12-factor: everything can be overridden with MENU_* environment variables)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MENU_", env_file=".env", extra="ignore")

    artifacts_dir: Path = Path("artifacts")
    # If artifacts_dir has no manifest.json, the bundle is downloaded from this Hugging Face repo (pinned revision).
    hf_repo: str | None = None
    hf_revision: str | None = None
    hf_repo_type: str = "model"

    enable_llm: bool = True
    enable_dense: bool = True
    llm_threads: int | None = None
    llm_device: str = "auto"
    chat_max_new_tokens: int = 120
    request_timeout_s: float = 60.0
    api_key: str | None = None  # if set, /v1/* requires the X-API-Key header
    mount_ui: bool = True

    log_level: str = "INFO"
    log_json: bool = True
    prediction_log: Path | None = None  # optional JSONL audit log of predictions
    monitoring_window: int = 500
    low_confidence: float = 0.4
    drift_oov_margin: float = 0.15  # alert when the rolling OOV rate exceeds the validation baseline by this much


@lru_cache
def get_settings() -> Settings:
    return Settings()
