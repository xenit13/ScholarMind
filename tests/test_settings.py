from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_mind.config import settings as config_settings
from scholar_mind.models.domain import IdeaNoveltyRequest


def test_zai_env_aliases_are_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.setenv("ZAI_BASE_URL", "https://example.invalid/v1")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.llm_api_key == "test-zai-key"
    assert settings.llm_base_url == "https://example.invalid/v1"


def test_scholarmind_env_overrides_zai_alias(monkeypatch, tmp_path):
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key")
    monkeypatch.setenv("ZAI_BASE_URL", "https://zai.invalid/v1")
    monkeypatch.setenv("SCHOLARMIND_LLM_API_KEY", "scoped-key")
    monkeypatch.setenv("SCHOLARMIND_LLM_BASE_URL", "https://scoped.invalid/v1")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.llm_api_key == "scoped-key"
    assert settings.llm_base_url == "https://scoped.invalid/v1"


def test_dotenv_aliases_are_loaded_without_leaking_process_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ZAI_API_KEY=dotenv-key\nZAI_BASE_URL=https://dotenv.invalid/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_settings, "ENV_FILE", env_file)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.llm_api_key == "dotenv-key"
    assert settings.llm_base_url == "https://dotenv.invalid/v1"


def test_dotenv_embedding_api_key_alias_is_loaded(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDING_API_KEY=dotenv-embedding-key\n", encoding="utf-8")
    monkeypatch.setattr(config_settings, "ENV_FILE", env_file)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_API_KEY", raising=False)
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.embedding_api_key == "dotenv-embedding-key"


def test_detect_root_dir_prefers_workdir_for_installed_package(monkeypatch, tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='scholar-mind'\n", encoding="utf-8")
    monkeypatch.delenv("SCHOLARMIND_ROOT_DIR", raising=False)

    root = config_settings._detect_root_dir(
        Path("/usr/local/lib/python3.11/site-packages/scholar_mind/config/settings.py"),
        cwd=tmp_path,
    )

    assert root == tmp_path.resolve()


def test_llm_timeout_env_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("SCHOLARMIND_LLM_REQUEST_TIMEOUT_SECONDS", "7.5")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.llm_request_timeout_seconds == 7.5


def test_environment_override_selects_matching_yaml(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "environment: development\nbootstrap_sample_data: true\n"
    )
    (config_dir / "development.yaml").write_text("reranker_enabled: false\n")
    (config_dir / "production.yaml").write_text("reranker_enabled: true\n")
    monkeypatch.setattr(config_settings, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("SCHOLARMIND_ENVIRONMENT", "production")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.environment == "production"
    assert settings.reranker_enabled is True


def test_production_defaults_to_local_qdrant_service(monkeypatch, tmp_path):
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("SCHOLARMIND_ENVIRONMENT", "production")
    monkeypatch.delenv("SCHOLARMIND_QDRANT_URL", raising=False)
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.qdrant_url == "http://127.0.0.1:6333"


def test_embedding_dimension_defaults_to_bge_m3_size():
    settings = config_settings.Settings(embedding_model="bge-m3")

    assert settings.resolved_embedding_dimension == 1024


def test_embedding_dimension_defaults_to_embedding_3_size():
    settings = config_settings.Settings(embedding_model="embedding-3")

    assert settings.resolved_embedding_dimension == 1024


def test_embedding_dimension_defaults_to_text_embedding_small_size():
    settings = config_settings.Settings(embedding_model="text-embedding-3-small")

    assert settings.resolved_embedding_dimension == 1536


def test_default_embedding_model_is_embedding_3():
    settings = config_settings.Settings()

    assert settings.embedding_model == "embedding-3"


def test_embedding_model_rejects_unsupported_value():
    with pytest.raises(ValidationError):
        config_settings.Settings(embedding_model="text-embedding-3-large")


def test_remote_reranker_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("SCHOLARMIND_RERANKER_PROVIDER", "remote")
    monkeypatch.setenv("SCHOLARMIND_RERANKER_BASE_URL", "http://host.docker.internal:18080")
    monkeypatch.setenv("SCHOLARMIND_RERANKER_REQUEST_TIMEOUT_SECONDS", "6.5")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.reranker_provider == "remote"
    assert settings.reranker_base_url == "http://host.docker.internal:18080"
    assert settings.reranker_request_timeout_seconds == 6.5


def test_memory_injection_defaults():
    settings = config_settings.Settings()

    assert settings.memory_top_k == 5
    assert settings.memory_min_similarity_score == 0.6
    assert settings.conditional_memory_injection is False
    assert settings.memory_structured_storage_enabled is True


def test_memory_yaml_conditional_memory_injection_is_loaded(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "environment: development\nconditional_memory_injection: false\n",
        encoding="utf-8",
    )
    (config_dir / "development.yaml").write_text("", encoding="utf-8")
    (config_dir / "memory.yaml").write_text(
        "memory:\n  conditional_memory_injection: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_settings, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.conditional_memory_injection is True


def test_memory_yaml_structured_storage_flag_is_loaded(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("environment: development\n", encoding="utf-8")
    (config_dir / "development.yaml").write_text("", encoding="utf-8")
    (config_dir / "memory.yaml").write_text(
        "memory:\n  structured_storage_enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_settings, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.memory_structured_storage_enabled is False


def test_memory_yaml_decay_settings_are_loaded(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("environment: development\n", encoding="utf-8")
    (config_dir / "development.yaml").write_text("", encoding="utf-8")
    (config_dir / "memory.yaml").write_text(
        "memory:\n"
        "  candidate_multiplier: 3\n"
        "  min_final_score: 0.12\n"
        "  decay:\n"
        "    enabled: false\n"
        "    archive_threshold: 0.02\n"
        "    access_boost_factor: 0.3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_settings, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.memory_candidate_multiplier == 3
    assert settings.memory_min_final_score == 0.12
    assert settings.memory_decay_enabled is False
    assert settings.memory_archive_threshold == 0.02
    assert settings.memory_access_boost_factor == 0.3


def test_rag_top_k_defaults_match_final_consumers():
    settings = config_settings.Settings()

    assert settings.final_citation_top_k == 4
    assert settings.idea_evidence_top_k == 10
    assert settings.cross_domain_candidate_top_k == 10
    assert settings.hybrid_candidate_multiplier == 4
    assert IdeaNoveltyRequest(idea="valid novelty idea", user_id="u").max_papers == 10


def test_memory_min_similarity_score_env_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config_settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("SCHOLARMIND_MEMORY_MIN_SIMILARITY_SCORE", "0.75")
    config_settings.get_settings.cache_clear()

    settings = config_settings.get_settings()

    assert settings.memory_min_similarity_score == 0.75
