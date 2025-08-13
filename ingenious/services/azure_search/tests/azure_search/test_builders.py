# -- coding: utf-8 --

"""
FILE TEST PLAN

    build_search_config_from_settings:
        maps fields from IngeniousSettings → SearchConfig
        validates missing azure_search_services[0], endpoint/key/index_name
        integrates _pick_models() result into SearchConfig (embedding/gen deployments + openai details)

    _pick_models:
        selection heuristics (embedding vs generation)
        requires deployment names for both
        warning path when single model reused
"""

import logging

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.builders import (
    ConfigError,
    _pick_models,
    build_search_config_from_settings,
)


def _settings(models, azure):
    s = IngeniousSettings.model_construct()
    s.models = models
    s.azure_search_services = [azure] if azure else None
    return s


def test_build_search_config_maps_and_validates(monkeypatch):
    models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="embed",
            api_key="K1",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="gpt-4o",
            deployment="chat",
            api_key="K2",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
    ]
    azure = AzureSearchSettings(
        service="svc",
        endpoint="https://search.windows.net",
        key="SKEY",
        index_name="idx",
    )
    cfg = build_search_config_from_settings(_settings(models, azure))
    assert cfg.search_endpoint == "https://search.windows.net"
    assert cfg.search_index_name == "idx"
    assert cfg.embedding_deployment_name == "embed"
    assert cfg.generation_deployment_name == "chat"
    assert cfg.openai_endpoint == "https://oai"
    assert cfg.openai_key.get_secret_value() in {"K2", "K1"}  # chosen from gen then emb


def test_build_search_config_errors():
    models = [
        ModelSettings(
            model="gpt-4o", deployment="chat", api_key="k", base_url="https://o"
        )
    ]
    with pytest.raises(ConfigError):
        build_search_config_from_settings(_settings(models, None))

    azure = AzureSearchSettings(service="svc", endpoint="", key="", index_name="")
    with pytest.raises(ConfigError):
        build_search_config_from_settings(_settings(models, azure))

    azure2 = AzureSearchSettings(
        service="svc", endpoint="https://s", key="k", index_name=""
    )
    with pytest.raises(ConfigError):
        build_search_config_from_settings(_settings(models, azure2))


def test_pick_models_selection_and_require_deployments(caplog):
    caplog.set_level(logging.WARNING)
    models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="emb",
            api_key="k",
            base_url="https://o",
        ),
    ]
    # Only one model present: should warn and reuse
    with pytest.raises(ConfigError):
        # deployment present but need both emb/gen (same), still valid deployments but function allows; however ensure
        # if we strip deployment it fails:
        _pick_models(
            _settings(
                [
                    ModelSettings(
                        model="gpt-4o", deployment="", api_key="k", base_url="https://o"
                    )
                ],
                AzureSearchSettings(
                    service="s", endpoint="https://e", key="k", index_name="i"
                ),
            )
        )
    # Single model with deployment for both roles should warn during selection path
    _ = _pick_models(
        _settings(
            models,
            AzureSearchSettings(
                service="s", endpoint="https://e", key="k", index_name="i"
            ),
        )
    )
    assert any("Single ModelSettings provided" in rec.message for rec in caplog.records)
