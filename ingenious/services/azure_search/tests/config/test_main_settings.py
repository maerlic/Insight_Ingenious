# tests/config/test_main_settings.py
import json

import pytest

from ingenious.config.main_settings import IngeniousSettings


def test_models_and_azure_search_from_json_env(monkeypatch):
    # Provide models via JSON string
    models_json = json.dumps(
        [
            {
                "model": "gpt-4o",
                "api_key": "k1",
                "base_url": "https://oai.example.com/",
                "deployment": "chat",
                "api_version": "2024-02-01",
            },
            {
                "model": "text-embedding-3-small",
                "api_key": "k1",
                "base_url": "https://oai.example.com/",
                "deployment": "embed",
                "api_version": "2024-02-01",
            },
        ]
    )
    azure_json = json.dumps(
        [
            {
                "service": "svc",
                "endpoint": "https://search.example.net",
                "key": "sk",
                "index_name": "idx",
                "use_semantic_ranking": True,
                "top_k_retrieval": 15,
            }
        ]
    )
    monkeypatch.setenv("INGENIOUS_MODELS", models_json)
    monkeypatch.setenv("INGENIOUS_AZURE_SEARCH_SERVICES", azure_json)

    settings = IngeniousSettings()
    assert len(settings.models) == 2
    assert settings.models[0].model == "gpt-4o"
    assert (
        settings.azure_search_services
        and settings.azure_search_services[0].index_name == "idx"
    )
    assert settings.azure_search_services[0].top_k_retrieval == 15


def test_models_and_azure_search_from_nested_env(monkeypatch):
    # Nested env for models
    monkeypatch.setenv("INGENIOUS_MODELS__0__MODEL", "gpt-4o")
    monkeypatch.setenv("INGENIOUS_MODELS__0__API_KEY", "k")
    monkeypatch.setenv("INGENIOUS_MODELS__0__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__0__DEPLOYMENT", "chat")
    monkeypatch.setenv("INGENIOUS_MODELS__0__API_VERSION", "2024-02-01")

    # Azure Search nested
    monkeypatch.setenv("INGENIOUS_AZURE_SEARCH_SERVICES__0__SERVICE", "svc")
    monkeypatch.setenv("INGENIOUS_AZURE_SEARCH_SERVICES__0__ENDPOINT", "https://s.net")
    monkeypatch.setenv("INGENIOUS_AZURE_SEARCH_SERVICES__0__KEY", "sk")
    monkeypatch.setenv("INGENIOUS_AZURE_SEARCH_SERVICES__0__INDEX_NAME", "idx")
    monkeypatch.setenv(
        "INGENIOUS_AZURE_SEARCH_SERVICES__0__USE_SEMANTIC_RANKING", "true"
    )
    monkeypatch.setenv("INGENIOUS_AZURE_SEARCH_SERVICES__0__TOP_K_RETRIEVAL", "25")

    settings = IngeniousSettings()
    assert settings.models[0].deployment == "chat"
    assert (
        settings.azure_search_services
        and settings.azure_search_services[0].endpoint == "https://s.net"
    )
    assert settings.azure_search_services[0].top_k_retrieval == 25


def test_invalid_port_and_log_level(monkeypatch):
    # minimal valid model
    monkeypatch.setenv("INGENIOUS_MODELS__0__MODEL", "gpt-4o")
    monkeypatch.setenv("INGENIOUS_MODELS__0__API_KEY", "k")
    monkeypatch.setenv("INGENIOUS_MODELS__0__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__0__DEPLOYMENT", "chat")

    # Invalid port
    monkeypatch.setenv("INGENIOUS_WEB_CONFIGURATION__PORT", "70000")
    with pytest.raises(ValueError):
        IngeniousSettings()

    # Fix port, break log level
    monkeypatch.delenv("INGENIOUS_WEB_CONFIGURATION__PORT", raising=False)
    monkeypatch.setenv("INGENIOUS_LOGGING__ROOT_LOG_LEVEL", "verbose")
    with pytest.raises(ValueError):
        IngeniousSettings()


def test_empty_models_rejected(monkeypatch):
    # No models env at all -> validator should reject
    with pytest.raises(Exception):
        IngeniousSettings()


def test_model_auth_validations_token_requires_api_key(monkeypatch):
    monkeypatch.setenv("INGENIOUS_MODELS__0__MODEL", "gpt-4o")
    monkeypatch.setenv("INGENIOUS_MODELS__0__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__0__DEPLOYMENT", "chat")
    monkeypatch.setenv("INGENIOUS_MODELS__0__AUTHENTICATION_METHOD", "token")
    # No API key set -> should fail
    with pytest.raises(ValueError):
        IngeniousSettings()


def test_model_auth_client_credentials_require_fields(monkeypatch):
    # Succeeds when client_id/secret & tenant provided (via env var or field)
    monkeypatch.setenv("INGENIOUS_MODELS__0__MODEL", "gpt-4o")
    monkeypatch.setenv("INGENIOUS_MODELS__0__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__0__DEPLOYMENT", "chat")
    monkeypatch.setenv(
        "INGENIOUS_MODELS__0__AUTHENTICATION_METHOD", "client_id_and_secret"
    )
    monkeypatch.setenv("INGENIOUS_MODELS__0__CLIENT_ID", "cid")
    monkeypatch.setenv("INGENIOUS_MODELS__0__CLIENT_SECRET", "csecret")
    # Provide tenant through AZURE_TENANT_ID env (allowed)
    monkeypatch.setenv("AZURE_TENANT_ID", "tenantX")
    settings = IngeniousSettings()
    assert settings.models[0].client_id == "cid"

    # Now missing tenant_id and AZURE_TENANT_ID -> should fail
    monkeypatch.setenv("INGENIOUS_MODELS__0__MODEL", "gpt-4o")
    monkeypatch.setenv("INGENIOUS_MODELS__0__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__0__DEPLOYMENT", "chat")
    monkeypatch.setenv(
        "INGENIOUS_MODELS__0__AUTHENTICATION_METHOD", "client_id_and_secret"
    )
    monkeypatch.setenv("INGENIOUS_MODELS__0__CLIENT_ID", "cid")
    monkeypatch.setenv("INGENIOUS_MODELS__0__CLIENT_SECRET", "csecret")
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    with pytest.raises(ValueError):
        IngeniousSettings()
