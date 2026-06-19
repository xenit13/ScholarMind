from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "config").is_dir()


def _detect_root_dir(anchor: Path | None = None, cwd: Path | None = None) -> Path:
    explicit = os.environ.get("SCHOLARMIND_ROOT_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()

    search_roots: list[Path] = []
    if anchor is not None:
        search_roots.extend(anchor.resolve().parents)
    if cwd is not None:
        search_roots.extend(cwd.resolve().parents)
        search_roots.insert(0, cwd.resolve())

    for candidate in search_roots:
        if _looks_like_project_root(candidate):
            return candidate
    return (anchor or Path(__file__)).resolve().parents[3]


ROOT_DIR = _detect_root_dir(Path(__file__), Path.cwd())
CONFIG_DIR = ROOT_DIR / "config"
ENV_FILE = ROOT_DIR / ".env"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCHOLARMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ScholarMind"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "sqlite:///data/sqlite/scholar_mind.db"
    checkpoint_database_url: str = "sqlite:///data/sqlite/checkpoints.db"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str | None = None
    qdrant_location: str = ":memory:"
    celery_task_always_eager: bool = False

    llm_provider: str = "openai_compatible"
    llm_reasoning_model: str = "glm-5.1"
    llm_light_model: str = "glm-4.5-air"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_request_timeout_seconds: float = 20.0
    llm_max_retries: int = 1

    embedding_model: Literal[
        "bge-m3", "baai/bge-m3", "embedding-3", "text-embedding-3-small"
    ] = "embedding-3"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    default_user_id: str = "local-user"
    message_context_window_tokens: int = 32768
    message_compact_threshold_ratio: float = 0.75
    memory_top_k: int = 5
    memory_min_similarity_score: float = Field(default=0.6, ge=0.0, le=1.0)
    memory_candidate_multiplier: int = Field(default=4, ge=1)
    memory_min_final_score: float = Field(default=0.05, ge=0.0)
    memory_decay_enabled: bool = True
    memory_archive_threshold: float = Field(default=0.01, ge=0.0)
    memory_access_boost_factor: float = Field(default=0.2, ge=0.0)
    memory_access_boost_cap: float = Field(default=1.5, ge=1.0)
    memory_explicit_keep_importance_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    conditional_memory_injection: bool = False
    memory_structured_storage_enabled: bool = True
    memory_consistency_audit_enabled: bool = True
    memory_consistency_audit_auto_fix_enabled: bool = True
    memory_consistency_audit_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    memory_consistency_audit_batch_size: int = Field(default=500, ge=1)

    log_dir: str = "data/message_logs"
    memory_root_dir: str = "data/memory"
    eval_root_dir: str = "data/eval"
    prompt_dir: str = "config/prompts"

    eval_enabled: bool = True

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @property
    def resolved_embedding_dimension(self) -> int:
        if self.embedding_model in {"bge-m3", "baai/bge-m3", "embedding-3"}:
            return 1024
        return 1536

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return ROOT_DIR / path


def _yaml_payload() -> dict[str, Any]:
    default = _load_yaml(CONFIG_DIR / "default.yaml")
    raw_env = {**_dotenv_values(), **os.environ}
    env_name = raw_env.get("SCHOLARMIND_ENVIRONMENT", default.get("environment", "development"))
    env_override = _load_yaml(CONFIG_DIR / f"{env_name}.yaml")
    merged = {**default, **_memory_yaml_payload(), **env_override}
    return merged


def _memory_yaml_payload() -> dict[str, Any]:
    payload = _load_yaml(CONFIG_DIR / "memory.yaml")
    memory = payload.get("memory", {})
    if not memory:
        return {}
    if not isinstance(memory, dict):
        raise ValueError(f"Invalid memory config payload in {CONFIG_DIR / 'memory.yaml'}")
    field_map = {
        "top_k": "memory_top_k",
        "min_similarity_score": "memory_min_similarity_score",
        "conditional_memory_injection": "conditional_memory_injection",
        "structured_storage_enabled": "memory_structured_storage_enabled",
        "candidate_multiplier": "memory_candidate_multiplier",
        "min_final_score": "memory_min_final_score",
        "root_dir": "memory_root_dir",
        "log_dir": "log_dir",
    }
    mapped = {target: memory[source] for source, target in field_map.items() if source in memory}
    decay = memory.get("decay", {})
    if isinstance(decay, dict):
        decay_field_map = {
            "enabled": "memory_decay_enabled",
            "archive_threshold": "memory_archive_threshold",
            "access_boost_factor": "memory_access_boost_factor",
            "access_boost_cap": "memory_access_boost_cap",
            "explicit_keep_importance_threshold": (
                "memory_explicit_keep_importance_threshold"
            ),
        }
        mapped.update(
            {
                target: decay[source]
                for source, target in decay_field_map.items()
                if source in decay
            }
        )
    consistency_audit = memory.get("consistency_audit", {})
    if isinstance(consistency_audit, dict):
        consistency_audit_field_map = {
            "enabled": "memory_consistency_audit_enabled",
            "auto_fix_enabled": "memory_consistency_audit_auto_fix_enabled",
            "min_confidence": "memory_consistency_audit_min_confidence",
            "batch_size": "memory_consistency_audit_batch_size",
        }
        mapped.update(
            {
                target: consistency_audit[source]
                for source, target in consistency_audit_field_map.items()
                if source in consistency_audit
            }
        )
    return mapped


def _dotenv_values() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        values[key.strip()] = cleaned
    return values


def _env_payload() -> dict[str, Any]:
    raw_env = {**_dotenv_values(), **os.environ}
    aliases = {
        "llm_api_key": ["SCHOLARMIND_LLM_API_KEY", "ZAI_API_KEY"],
        "llm_base_url": ["SCHOLARMIND_LLM_BASE_URL", "ZAI_BASE_URL"],
        "embedding_api_key": ["SCHOLARMIND_EMBEDDING_API_KEY", "EMBEDDING_API_KEY"],
        "embedding_base_url": ["SCHOLARMIND_EMBEDDING_BASE_URL"],
    }
    payload: dict[str, Any] = {}
    for field_name in Settings.model_fields:
        candidate_keys = aliases.get(field_name, [f"SCHOLARMIND_{field_name.upper()}"])
        for env_key in candidate_keys:
            if env_key in raw_env:
                payload[field_name] = raw_env[env_key]
                break
    return payload


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(**{**_yaml_payload(), **_env_payload()})
