# tests/config/test_validators_extra.py

import logging

import pytest
from pydantic import ValidationError

from ingenious.common.enums import AuthenticationMethod
from ingenious.config import get_config
from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import (
    AzureSearchSettings,
    LoggingSettings,
    ModelSettings,
    WebSettings,
)

# ─────────────────────────────────────────────────────────────────────────────
# ModelSettings
# ─────────────────────────────────────────────────────────────────────────────


def test_modelsettings_token_requires_api_key():
    # Missing API key with TOKEN auth → should raise ValidationError
    with pytest.raises(ValidationError):
        ModelSettings(
            model="gpt-4o",
            base_url="https://oai.example.com",
            deployment="chat",
            authentication_method=AuthenticationMethod.TOKEN,
            api_key="",  # required for TOKEN
        )

    # With API key present → should construct fine
    ok = ModelSettings(
        model="gpt-4o",
        base_url="https://oai.example.com",
        deployment="chat",
        authentication_method=AuthenticationMethod.TOKEN,
        api_key="secret",
    )
    assert ok.api_key == "secret"


def test_modelsettings_client_id_and_secret_require_all_fields_or_env(monkeypatch):
    # Tenant ID missing on the object, but present via AZURE_TENANT_ID env → allowed
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-123")

    ms = ModelSettings(
        model="gpt-4o",
        base_url="https://oai.example.com",
        deployment="chat",
        authentication_method=AuthenticationMethod.CLIENT_ID_AND_SECRET,
        client_id="cid",
        client_secret="csecret",
        tenant_id="",  # intentionally missing — satisfied via env var
    )
    assert ms.client_id == "cid"
    assert ms.client_secret == "csecret"
    # Clean up (pytest monkeypatch auto-cleans, but keep intent explicit)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)


@pytest.mark.parametrize(
    "base_url,should_raise",
    [
        ("https://ok.example.com", False),
        ("http://ok.example.com", False),
        ("ftp://bad.example.com", True),  # invalid scheme
        ("PLACEHOLDER", True),  # forbidden placeholder pattern
        ("", False),  # empty is allowed (validator only checks when value is provided)
    ],
)
def test_modelsettings_base_url_validation(base_url, should_raise):
    kwargs = dict(
        model="gpt-4o",
        deployment="chat",
        api_key="",  # not using TOKEN auth here
    )
    if should_raise:
        with pytest.raises(ValidationError):
            ModelSettings(base_url=base_url, **kwargs)
    else:
        ms = ModelSettings(base_url=base_url, **kwargs)
        assert ms.base_url == base_url


# ─────────────────────────────────────────────────────────────────────────────
# LoggingSettings
# ─────────────────────────────────────────────────────────────────────────────


def test_logging_level_normalization_and_reject_unknown():
    ls = LoggingSettings(root_log_level="DEBUG", log_level="WARNING")
    assert ls.root_log_level == "debug"
    assert ls.log_level == "warning"

    with pytest.raises(ValidationError):
        LoggingSettings(root_log_level="verbose")  # unknown level


# ─────────────────────────────────────────────────────────────────────────────
# WebSettings
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_websettings_port_bounds_invalid(port):
    with pytest.raises(ValidationError):
        WebSettings(port=port)


@pytest.mark.parametrize("port", [1, 80, 443, 65535])
def test_websettings_port_bounds_valid(port):
    ws = WebSettings(port=port)
    assert ws.port == port


# ─────────────────────────────────────────────────────────────────────────────
# IngeniousSettings field parsing (dict → list)
# ─────────────────────────────────────────────────────────────────────────────


def test_ingenioussettings_parse_models_from_nested_env_dict():
    # Provide 'models' as a dict that mimics nested env-structure keys
    models_dict = {
        "1": {
            "model": "gpt-4o",
            "api_type": "rest",
            "api_version": "2024-02-01",
            "deployment": "chat",
            "base_url": "https://oai.example.com",
        },
        "0": {
            "model": "text-embedding-3-small",
            "api_type": "rest",
            "api_version": "2024-02-01",
            "deployment": "embed",
            "base_url": "https://oai.example.com",
        },
    }
    s = IngeniousSettings(models=models_dict)
    assert len(s.models) == 2
    # Keys are sorted; "0" should come before "1"
    assert s.models[0].model == "text-embedding-3-small"
    assert s.models[1].model == "gpt-4o"


def test_ingenioussettings_parse_azure_search_from_nested_env_dict():
    models_dict = {
        "0": {
            "model": "gpt-4o",
            "api_type": "rest",
            "api_version": "2024-02-01",
            "deployment": "chat",
            "base_url": "https://oai.example.com",
        }
    }
    azure_dict = {
        "0": {
            "service": "svc",
            "endpoint": "https://search.example.net",
            "key": "sk",
            "index_name": "idx",
            "use_semantic_ranking": True,
            "top_k_retrieval": 15,
        }
    }
    s = IngeniousSettings(models=models_dict, azure_search_services=azure_dict)
    assert s.azure_search_services is not None
    assert len(s.azure_search_services) == 1
    az0 = s.azure_search_services[0]
    assert isinstance(az0, AzureSearchSettings.__class__) or hasattr(az0, "index_name")
    assert az0.index_name == "idx"
    assert az0.top_k_retrieval == 15
    assert az0.use_semantic_ranking is True


# ─────────────────────────────────────────────────────────────────────────────
# get_config() logging + re-raise on validation error
# ─────────────────────────────────────────────────────────────────────────────


def test_get_config_logs_and_reraises_on_validation_error(monkeypatch, caplog):
    # Patch logger factory to return a standard logger we can capture with caplog
    import logging as _logging

    def fake_get_logger(_name):
        return _logging.getLogger("ing-config-test")

    monkeypatch.setattr(
        "ingenious.core.structured_logging.get_logger",
        fake_get_logger,
        raising=True,
    )

    # Force IngeniousSettings() to raise in constructor so get_config logs and re-raises
    class BoomSettings:
        def __init__(self, *a, **k):
            raise RuntimeError("boom-config")

    monkeypatch.setattr(
        "ingenious.config.IngeniousSettings", BoomSettings, raising=True
    )

    with caplog.at_level(logging.ERROR, logger="ing-config-test"):
        with pytest.raises(RuntimeError, match="boom-config"):
            get_config()

    # Ensure our error message was logged
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Failed to load configuration: boom-config" in m for m in messages)
