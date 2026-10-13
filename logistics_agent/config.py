"""Centralised, validated configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: Optional[str] = None

    fast_model: str = "gpt-4o-mini"
    validator_model: str = "gpt-4o"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    policy_file: Path = Path("data/logistics_policy_rulebook.txt")
    test_cases_file: Path = Path("data/test_cases_curated.csv")
    vector_store_dir: Optional[Path] = None

    confidence_threshold: float = 0.6
    dimension_threshold: float = 0.6
    overall_threshold: float = 0.7
    deliver_fallback_score: float = 0.6

    max_searches: int = 3
    max_iterations: int = 8
    max_validation_failures: int = 2
    recursion_limit: int = 50
    policy_snippet_chars: int = 300

    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required")
        for name in ("confidence_threshold", "dimension_threshold",
                     "overall_threshold", "deliver_fallback_score"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1")
        for name in ("max_searches", "max_iterations", "max_validation_failures"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Settings":
        """Build settings from the environment (and an optional .env file)."""
        load_dotenv(env_file)
        store_dir = os.getenv("VECTOR_STORE_DIR")
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or None,
            fast_model=os.getenv("FAST_MODEL", cls.fast_model),
            validator_model=os.getenv("VALIDATOR_MODEL", cls.validator_model),
            llm_timeout_seconds=_get_float("LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            llm_max_retries=_get_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            policy_file=Path(os.getenv("POLICY_FILE", str(cls.policy_file))),
            test_cases_file=Path(os.getenv("TEST_CASES_FILE", str(cls.test_cases_file))),
            vector_store_dir=Path(store_dir) if store_dir else None,
            confidence_threshold=_get_float("CONFIDENCE_THRESHOLD", cls.confidence_threshold),
            dimension_threshold=_get_float("DIMENSION_THRESHOLD", cls.dimension_threshold),
            overall_threshold=_get_float("OVERALL_THRESHOLD", cls.overall_threshold),
            max_searches=_get_int("MAX_SEARCHES", cls.max_searches),
            max_iterations=_get_int("MAX_ITERATIONS", cls.max_iterations),
            max_validation_failures=_get_int("MAX_VALIDATION_FAILURES", cls.max_validation_failures),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )
