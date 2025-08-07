import pytest
from pydantic import ValidationError, SecretStr

# Imports rely on the structure defined in conftest.py
try:
    from ingenious.services.azure_search.config import SearchConfig, DEFAULT_DAT_PROMPT
except ImportError:
    # If imports fail, we skip the module
    pytest.skip("Could not import SearchConfig", allow_module_level=True)


def test_search_config_valid(config: SearchConfig):
    """Test that the fixture provides a valid configuration."""
    assert config.search_endpoint.startswith("https://")
    assert config.openai_endpoint.startswith("https://")
    assert config.search_key.get_secret_value() == "test_search_key_12345"
    assert isinstance(config.openai_key, SecretStr)
    assert config.top_k_retrieval > 0
    assert config.use_semantic_ranking is True
    assert config.semantic_configuration_name is not None

def test_search_config_missing_required_fields():
    """Test validation errors when required fields are missing."""
    minimal_config = {
        "search_endpoint": "http://localhost",
        "search_key": SecretStr("key1"),
        "search_index_name": "index",
        # Missing OpenAI config and deployment names
    }
    with pytest.raises(ValidationError) as excinfo:
        SearchConfig(**minimal_config)

    errors = excinfo.value.errors()
    # Expect errors for openai_endpoint, openai_key, embedding_deployment_name, generation_deployment_name
    assert len(errors) >= 4
    error_locs = [e['loc'][0] for e in errors]
    assert 'openai_endpoint' in error_locs
    assert 'embedding_deployment_name' in error_locs

def test_search_config_immutability(config: SearchConfig):
    """Test that the configuration object is frozen (immutable)."""
    # Pydantic V2 uses ValidationError (FrozenError) for modification attempts
    with pytest.raises(ValidationError):
        config.top_k_retrieval = 100

def test_default_dat_prompt_structure():
    """Verify the structure and key instructions of the default DAT prompt."""
    assert "System:" in DEFAULT_DAT_PROMPT
    assert "Scoring Criteria:" in DEFAULT_DAT_PROMPT
    assert "Direct Hit -> 5 points" in DEFAULT_DAT_PROMPT
    assert "Completely Off-Track -> 0 points" in DEFAULT_DAT_PROMPT
    assert "Output Format:" in DEFAULT_DAT_PROMPT
    assert "Respond ONLY with two integers separated by a single space." in DEFAULT_DAT_PROMPT

def test_search_config_defaults():
    """Test that default values are correctly applied when not provided."""
    base_config = {
        "search_endpoint": "http://localhost",
        "search_key": SecretStr("key1"),
        "search_index_name": "index",
        "openai_endpoint": "http://localhost/oai",
        "openai_key": SecretStr("key2"),
        "embedding_deployment_name": "embed",
        "generation_deployment_name": "gen",
    }
    config = SearchConfig(**base_config)

    # Check defaults defined in the SearchConfig model
    assert config.top_k_retrieval == 20
    assert config.use_semantic_ranking is True
    assert config.top_n_final == 5
    assert config.id_field == "id"
    assert config.content_field == "content"
    assert config.vector_field == "vector"
    assert config.openai_version == "2024-02-01"
    assert config.dat_prompt == DEFAULT_DAT_PROMPT
    assert config.semantic_configuration_name is None