# tests/config/test_models_validation.py
import json

import pytest

from ingenious.config.main_settings import IngeniousSettings


def test_parse_models_json_and_nested_mix(monkeypatch):
    # Configuration via JSON string should override nested variables
    models_json = json.dumps(
        [
            {
                "model": "gpt-4o",
                "api_key": "k",
                "base_url": "https://oai/",
                "deployment": "chat",
            }
        ]
    )
    monkeypatch.setenv("INGENIOUS_MODELS", models_json)
    # These nested variables should be ignored
    monkeypatch.setenv("INGENIOUS_MODELS__1__MODEL", "text-embedding-3-small")
    monkeypatch.setenv("INGENIOUS_MODELS__1__API_KEY", "k2")
    monkeypatch.setenv("INGENIOUS_MODELS__1__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__1__DEPLOYMENT", "embed")

    settings = IngeniousSettings()
    # Expect only the model from the JSON string
    assert len(settings.models) == 1
    assert settings.models[0].deployment == "chat"


def test_web_settings_port_range(monkeypatch):
    # minimal valid model
    monkeypatch.setenv("INGENIOUS_MODELS__0__MODEL", "gpt-4o")
    monkeypatch.setenv("INGENIOUS_MODELS__0__API_KEY", "k")
    monkeypatch.setenv("INGENIOUS_MODELS__0__BASE_URL", "https://oai/")
    monkeypatch.setenv("INGENIOUS_MODELS__0__DEPLOYMENT", "chat")

    # lower bound
    monkeypatch.setenv("INGENIOUS_WEB_CONFIGURATION__PORT", "0")
    with pytest.raises(ValueError):
        IngeniousSettings()

    monkeypatch.setenv("INGENIOUS_WEB_CONFIGURATION__PORT", "65536")
    with pytest.raises(ValueError):
        IngeniousSettings()

    monkeypatch.setenv("INGENIOUS_WEB_CONFIGURATION__PORT", "443")
    s = IngeniousSettings()
    assert s.web_configuration.port == 443
