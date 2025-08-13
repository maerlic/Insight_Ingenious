# tests/azure_search/test_builders_multi_model_selection.py
# -*- coding: utf-8 -*-
import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.builders import ConfigError, _pick_models


def _settings(models, azure=None):
    s = IngeniousSettings.model_construct()
    s.models = models
    s.azure_search_services = [azure] if azure else []
    return s


def test_pick_models_first_match_deterministic():
    models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="emb-1",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="text-embedding-3-large",
            deployment="emb-2",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="gpt-4o",
            deployment="chat-1",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="gpt-35-turbo",
            deployment="chat-2",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
    ]
    azure = AzureSearchSettings(
        service="svc", endpoint="https://s", key="sk", index_name="idx"
    )

    picked = _pick_models(_settings(models, azure))

    # Accept either tuple or object return shapes
    if isinstance(picked, tuple):
        *_, emb_dep, gen_dep = picked
    else:
        emb_dep = getattr(picked, "embedding_deployment", None)
        gen_dep = getattr(picked, "generation_deployment", None)

    assert emb_dep == "emb-1"  # first embedding wins
    assert gen_dep == "chat-1"  # first GPT/4o-ish wins


def test_pick_models_requires_any_valid_candidates():
    models = [
        ModelSettings(
            model="other-model",
            deployment="other",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        )
    ]
    azure = AzureSearchSettings(
        service="svc", endpoint="https://s", key="sk", index_name="idx"
    )
    with pytest.raises(ConfigError):
        _pick_models(_settings(models, azure))
