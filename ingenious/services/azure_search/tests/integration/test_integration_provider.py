# file: ingenious/services/azure_search/tests/test_integration_provider.py

import os

import pytest

pytestmark = pytest.mark.azure_integration

REQUIRED_ENV = [
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_GENERATION_DEPLOYMENT",
]

# Optional; if not present we default to a commonly used stable version.
DEFAULT_OPENAI_API_VERSION = "2024-06-01"


def _require_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Skipping: missing env vars: {', '.join(missing)}")
    return {k: os.environ[k] for k in keys}


def _lazy_imports():
    """
    Import inside the test so collection doesn't fail if paths differ.
    Adjust these to your project’s actual module paths if needed:
      - AzureSearchProvider: ingenious.services.azure_search.provider
      - build_search_config_from_settings: ingenious.services.azure_search.builders
      - IngeniousSettings: ingenious.settings
    """
    try:
        from ingenious.services.azure_search.provider import AzureSearchProvider
    except Exception as e:
        pytest.skip(f"Cannot import AzureSearchProvider: {e}")

    try:
        from ingenious.services.azure_search.builders import (
            build_search_config_from_settings,
        )
    except Exception as e:
        pytest.skip(f"Cannot import build_search_config_from_settings: {e}")

    try:
        from ingenious.settings import IngeniousSettings
    except Exception as e:
        pytest.skip(f"Cannot import IngeniousSettings: {e}")

    return AzureSearchProvider, build_search_config_from_settings, IngeniousSettings


@pytest.mark.asyncio
async def test_end_to_end_provider_with_real_service_no_semantic():
    """
    Smoke test the AzureSearchProvider against a real Azure Search + Azure OpenAI setup,
    with semantic ranking disabled. This catches auth/index/schema drift.
    """
    env = _require_env(REQUIRED_ENV)
    AzureSearchProvider, build_from_settings, IngeniousSettings = _lazy_imports()

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_OPENAI_API_VERSION)

    # Build a minimal, valid settings object that your builder understands.
    # If your settings schema differs, tweak these keys accordingly.
    settings = IngeniousSettings(
        models=[
            {
                "provider": "azure",
                "family": "openai",
                "type": "embedding",
                "api_base": env["AZURE_OPENAI_ENDPOINT"],
                "api_key": env["AZURE_OPENAI_KEY"],
                "api_version": api_version,
                "deployment": env["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            },
            {
                "provider": "azure",
                "family": "openai",
                "type": "chat",
                "api_base": env["AZURE_OPENAI_ENDPOINT"],
                "api_key": env["AZURE_OPENAI_KEY"],
                "api_version": api_version,
                "deployment": env["AZURE_OPENAI_GENERATION_DEPLOYMENT"],
            },
        ],
        azure_search_services=[
            {
                "endpoint": env["AZURE_SEARCH_ENDPOINT"],
                "api_key": env["AZURE_SEARCH_KEY"],
                "index_name": env["AZURE_SEARCH_INDEX_NAME"],
                # Explicitly disable semantic ranking for this test
                "enable_semantic_ranking": False,
            }
        ],
    )

    config = build_from_settings(settings)
    provider = AzureSearchProvider(config)

    try:
        results = await provider.retrieve("smoke query", top_k=1)
        assert isinstance(results, list)
    finally:
        # Make sure we close network clients even if the assertion fails.
        await provider.close()


@pytest.mark.asyncio
async def test_end_to_end_provider_with_real_service_with_semantic():
    """
    Same smoke test but with semantic ranking enabled.
    Skips unless AZURE_SEARCH_SEMANTIC_CONFIG is set (must exist in your index).
    """
    env = _require_env(REQUIRED_ENV)
    semantic_name = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")
    if not semantic_name:
        pytest.skip("Skipping: AZURE_SEARCH_SEMANTIC_CONFIG not set")

    AzureSearchProvider, build_from_settings, IngeniousSettings = _lazy_imports()

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_OPENAI_API_VERSION)

    settings = IngeniousSettings(
        models=[
            {
                "provider": "azure",
                "family": "openai",
                "type": "embedding",
                "api_base": env["AZURE_OPENAI_ENDPOINT"],
                "api_key": env["AZURE_OPENAI_KEY"],
                "api_version": api_version,
                "deployment": env["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            },
            {
                "provider": "azure",
                "family": "openai",
                "type": "chat",
                "api_base": env["AZURE_OPENAI_ENDPOINT"],
                "api_key": env["AZURE_OPENAI_KEY"],
                "api_version": api_version,
                "deployment": env["AZURE_OPENAI_GENERATION_DEPLOYMENT"],
            },
        ],
        azure_search_services=[
            {
                "endpoint": env["AZURE_SEARCH_ENDPOINT"],
                "api_key": env["AZURE_SEARCH_KEY"],
                "index_name": env["AZURE_SEARCH_INDEX_NAME"],
                "enable_semantic_ranking": True,
                "semantic_configuration_name": semantic_name,
            }
        ],
    )

    config = build_from_settings(settings)
    provider = AzureSearchProvider(config)

    try:
        results = await provider.retrieve("smoke query", top_k=1)
        assert isinstance(results, list)
    finally:
        await provider.close()
