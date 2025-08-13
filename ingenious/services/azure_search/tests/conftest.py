# -- coding: utf-8 --

"""
COVERAGE PLAN

    Provide common fixtures, dummies, and precise monkeypatching to keep tests offline.
    Replace Azure/OpenAI clients with AsyncMock instances returned by our own factories.
    Patch QueryType/VectorizedQuery used by retrieval module to avoid azure SDK import needs.
    Build a canonical SearchConfig with dummy secrets and toggle variant without semantic ranking.

    Helpers:
        AsyncIter for async iteration in semantic reranking paths.
        make_search_client / make_async_openai_client monkeypatch targets.

    Ensure components/pipeline/cli import paths work whether package is editable or installed.
"""

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

# Public model under test
from ingenious.services.azure_search.config import DEFAULT_DAT_PROMPT, SearchConfig

# --- Async iterator helper ---------------------------------------------------


class AsyncIter:
    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


# --- Dummy Azure SDK model stand-ins -----------------------------------------


class DummyQueryType:
    SIMPLE = "simple"
    SEMANTIC = "semantic"


class DummyVectorizedQuery:
    def __init__(
        self,
        *,
        vector: List[float],
        k_nearest_neighbors: int,
        fields: str,
        exhaustive: bool = True,
    ):
        self.vector = vector
        self.k_nearest_neighbors = k_nearest_neighbors
        self.fields = fields
        self.exhaustive = exhaustive


# --- Core config fixtures -----------------------------------------------------


@pytest.fixture
def config() -> SearchConfig:
    """Valid SearchConfig for most tests."""
    return SearchConfig(
        search_endpoint="https://unit-search.windows.net",
        search_key=SecretStr("search_key"),
        search_index_name="unit-index",
        semantic_configuration_name="sem-config",
        openai_endpoint="https://unit-openai.azure.com",
        openai_key=SecretStr("openai_key"),
        openai_version="2024-02-01",
        embedding_deployment_name="embed-deploy",
        generation_deployment_name="chat-deploy",
        top_k_retrieval=10,
        use_semantic_ranking=True,
        top_n_final=3,
        id_field="id",
        content_field="content",
        vector_field="vector",
        dat_prompt=DEFAULT_DAT_PROMPT,
    )


@pytest.fixture
def config_no_semantic(config: SearchConfig) -> SearchConfig:
    """Variant with semantic ranking disabled."""
    data = config.model_dump(exclude={"search_key", "openai_key"})
    data["use_semantic_ranking"] = False
    data["semantic_configuration_name"] = None
    data["search_key"] = config.search_key
    data["openai_key"] = config.openai_key
    return SearchConfig(**data)


# --- Global monkeypatches for external deps ----------------------------------


@pytest.fixture(autouse=True)
def patch_external_sdks(monkeypatch):
    """
    - Patch azure.search.documents.models symbols referenced directly by modules.
    - Patch client factory functions used by components/pipeline.
    - Provide OpenAI-like Async client with embeddings/chat APIs.
    """
    # Patch model symbols where modules import them
    monkeypatch.setitem(globals(), "DummyQueryType", DummyQueryType)
    monkeypatch.setitem(globals(), "DummyVectorizedQuery", DummyVectorizedQuery)

    # Retrieval module uses these names at import time
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "0")  # make asyncio errors clearer

    # Patch azure.search.documents.models.QueryType & VectorizedQuery to our dummies
    for target in [
        "ingenious.services.azure_search.components.retrieval.QueryType",
        "ingenious.services.azure_search.components.pipeline.QueryType",
    ]:
        try:
            monkeypatch.setattr(target, DummyQueryType, raising=False)
        except Exception:
            pass
    try:
        monkeypatch.setattr(
            "ingenious.services.azure_search.components.retrieval.VectorizedQuery",
            DummyVectorizedQuery,
            raising=False,
        )
    except Exception:
        pass

    # Create shared async OpenAI mock with embeddings & chat
    openai_client = AsyncMock(name="AsyncAzureOpenAI")
    openai_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.01] * 3)])
    )
    openai_client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="3 4"))]
        )
    )

    # Create shared search client mock with .search() async iterator and .close()
    search_client = AsyncMock(name="SearchClient")
    search_client.search = AsyncMock(return_value=AsyncIter([]))
    search_client.close = AsyncMock()

    # Patch factory functions to return our shared clients
    monkeypatch.setattr(
        "ingenious.services.azure_search.client_init.make_async_openai_client",
        lambda cfg: openai_client,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.client_init.make_search_client",
        lambda cfg: search_client,
        raising=False,
    )

    # Components import factories dynamically; also patch potential direct paths
    for mod in [
        "ingenious.services.azure_search.components.retrieval",
        "ingenious.services.azure_search.components.fusion",
        "ingenious.services.azure_search.components.generation",
        "ingenious.services.azure_search.pipeline",
    ]:
        try:
            monkeypatch.setattr(
                f"{mod}.make_async_openai_client",
                lambda cfg: openai_client,
                raising=False,
            )
        except Exception:
            pass
        try:
            monkeypatch.setattr(
                f"{mod}.make_search_client", lambda cfg: search_client, raising=False
            )
        except Exception:
            pass

    # Expose to tests that need them
    yield


# Utility fixtures re-exposed for tests that want them


@pytest.fixture
def async_iter():  # noqa: D401
    """Return AsyncIter helper class."""
    return AsyncIter
