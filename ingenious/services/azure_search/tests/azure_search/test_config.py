# -- coding: utf-8 --

"""
FILE TEST PLAN

    Validate SearchConfig required fields and defaults.
    Confirm immutability (frozen model).
    Sanity-check DEFAULT_DAT_PROMPT structure and key guidance.
"""

import pytest
from pydantic import SecretStr, ValidationError

from ingenious.services.azure_search.config import DEFAULT_DAT_PROMPT, SearchConfig


def test_search_config_valid(config: SearchConfig):
    assert config.search_endpoint.startswith("https://")
    assert config.openai_endpoint.startswith("https://")
    assert isinstance(config.search_key, SecretStr)
    assert isinstance(config.openai_key, SecretStr)
    assert config.embedding_deployment_name
    assert config.generation_deployment_name
    assert config.openai_version == "2024-02-01"
    assert config.use_semantic_ranking is True


def test_search_config_missing_required_fields():
    data = dict(
        search_endpoint="http://localhost",
        search_key=SecretStr("x"),
        search_index_name="idx",
    )
    with pytest.raises(ValidationError) as e:
        SearchConfig(**data)
    locs = {tuple(err["loc"]) for err in e.value.errors()}
    assert ("openai_endpoint",) in locs
    assert ("openai_key",) in locs
    assert ("embedding_deployment_name",) in locs
    assert ("generation_deployment_name",) in locs


def test_search_config_defaults_minimal_ok():
    cfg = SearchConfig(
        search_endpoint="http://s",
        search_key=SecretStr("a"),
        search_index_name="i",
        openai_endpoint="http://o",
        openai_key=SecretStr("b"),
        embedding_deployment_name="e",
        generation_deployment_name="g",
    )
    assert cfg.top_k_retrieval == 20
    assert cfg.top_n_final == 5
    assert cfg.id_field == "id"
    assert cfg.content_field == "content"
    assert cfg.vector_field == "vector"
    assert cfg.dat_prompt == DEFAULT_DAT_PROMPT
    assert cfg.semantic_configuration_name is None


def test_search_config_is_frozen(config: SearchConfig):
    with pytest.raises(ValidationError):
        config.top_k_retrieval = 99  # type: ignore[misc]


def test_default_dat_prompt_has_key_sections():
    s = DEFAULT_DAT_PROMPT
    assert "System:" in s
    assert "Scoring Criteria" in s
    assert "Direct Hit -> 5 points" in s
    assert "Completely Off-Track -> 0 points" in s
    assert "Respond ONLY with two integers" in s
