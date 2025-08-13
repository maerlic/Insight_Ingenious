# -- coding: utf-8 --
"""
Provider tests for AzureSearchProvider.

Covers:
  - retrieve(): semantic happy path (merging + cleaning)
  - retrieve(): fallback when IDs are missing (no semantic call)
  - answer(): delegation to pipeline.get_answer()
  - close(): supports both sync and async close() on dependencies
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider


def _make_settings() -> IngeniousSettings:
    """Build a lightweight IngeniousSettings object suitable for provider init."""
    s = IngeniousSettings.model_construct()
    s.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="embed",
            api_key="K",
            base_url="https://oai.example.com",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="gpt-4o",
            deployment="chat",
            api_key="K",
            base_url="https://oai.example.com",
            api_version="2024-02-01",
        ),
    ]
    s.azure_search_services = [
        AzureSearchSettings(
            service="svc",
            endpoint="https://search.example.net",
            key="SK",
            index_name="idx",
        )
    ]
    return s


# --- tests -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_retrieve_semantic_happy_path(monkeypatch, async_iter):
    """
    - Semantic reranker returns results; internal merge sets _final_score and updates content.
    - Public retrieve() returns cleaned docs (no internal fields, preserved updated content).
    """
    settings = _make_settings()

    # Mock pipeline components
    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(
        return_value=[{"id": "A", "_retrieval_score": 1.0}]
    )
    mock_pipeline.retriever.search_vector = AsyncMock(
        return_value=[{"id": "A", "_retrieval_score": 0.9, "vector": [0.1]}]
    )
    mock_pipeline.fuser.fuse = AsyncMock(
        return_value=[
            {"id": "A", "_fused_score": 0.8, "@search.score": 1.0, "vector": [0.2]}
        ]
    )

    # Reranker returns a new score and updated content
    fake_rerank_client = MagicMock()
    fake_rerank_client.search = AsyncMock(
        return_value=async_iter(
            [{"id": "A", "@search.reranker_score": 2.5, "content": "Alpha"}]
        )
    )

    # Wire seams
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_rerank_client,
        raising=False,
    )
    # Provide a QueryType shim
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    prov = AzureSearchProvider(settings)

    # Internal: _semantic_rerank should set _final_score and merge content
    fused = [{"id": "A", "_fused_score": 0.8}]
    internal = await prov._semantic_rerank("q", fused)
    assert internal and internal[0]["_final_score"] == 2.5
    assert internal[0]["content"] == "Alpha"

    # Public: retrieve() should return cleaned version
    out = await prov.retrieve("q", top_k=1)
    assert len(out) == 1
    assert out[0]["id"] == "A"
    assert out[0]["content"] == "Alpha"
    # cleaned fields
    for k in (
        "_fused_score",
        "_final_score",
        "@search.score",
        "@search.reranker_score",
        "vector",
    ):
        assert k not in out[0]

    await prov.close()


@pytest.mark.asyncio
async def test_provider_retrieve_semantic_error_fallback(monkeypatch):
    """
    Fallback path when semantic cannot run due to missing IDs:
      - No IDs → provider skips semantic rerank and uses fused scores.
      - Returned docs are cleaned for caller.
    NOTE: Provider does not catch reranker exceptions; this test covers the supported
    fallback mode (missing IDs), which is the resilient path implemented in code.
    """
    settings = _make_settings()

    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(
        return_value=[{"X": "no-id", "_retrieval_score": 1.0}]
    )
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[])
    fused = [{"X": "no-id", "_fused_score": 0.42, "content": "C"}]
    mock_pipeline.fuser.fuse = AsyncMock(return_value=fused)

    fake_rerank_client = MagicMock()
    # If semantic was attempted, we would raise — but it should NOT be called because IDs are missing.
    fake_rerank_client.search = AsyncMock(side_effect=RuntimeError("rerank failed"))

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_rerank_client,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    out = await prov.retrieve("q", top_k=1)

    # Should return 1 doc (cleaned), and semantic client wasn't used
    assert len(out) == 1
    doc = out[0]
    for k in (
        "_fused_score",
        "_final_score",
        "@search.score",
        "@search.reranker_score",
    ):
        assert k not in doc
    fake_rerank_client.search.assert_not_called()
    await prov.close()


@pytest.mark.asyncio
async def test_provider_answer_delegates_pipeline(monkeypatch, async_iter):
    """
    provider.answer() should delegate to pipeline.get_answer() and return its result.
    """
    settings = _make_settings()

    mock_pipeline = MagicMock()
    mock_pipeline.get_answer = AsyncMock(
        return_value={"answer": "A", "source_chunks": [{"id": "S"}]}
    )

    fake_rerank_client = MagicMock()
    fake_rerank_client.search = AsyncMock(return_value=async_iter([]))

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_rerank_client,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    ans = await prov.answer("what?")
    assert ans["answer"] == "A"
    await prov.close()


@pytest.mark.parametrize("async_close", [True, False])
@pytest.mark.asyncio
async def test_provider_close_tolerates_sync_or_async(
    monkeypatch, async_close: bool, async_iter
):
    """
    provider.close() should work whether dependencies expose sync or async close().
    """
    settings = _make_settings()

    # Pipeline with close() sync or async
    if async_close:
        mock_pipeline = MagicMock()
        mock_pipeline.close = AsyncMock()
    else:
        mock_pipeline = MagicMock()
        mock_pipeline.close = MagicMock(return_value=None)

    # Rerank client with close() sync or async
    if async_close:
        fake_rerank_client = MagicMock()
        fake_rerank_client.search = AsyncMock(return_value=async_iter([]))
        fake_rerank_client.close = AsyncMock()
    else:
        fake_rerank_client = MagicMock()
        fake_rerank_client.search = AsyncMock(return_value=async_iter([]))
        fake_rerank_client.close = MagicMock(return_value=None)

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_rerank_client,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    await prov.close()

    # Verify the appropriate close path was used without raising
    if async_close:
        mock_pipeline.close.assert_awaited()
        fake_rerank_client.close.assert_awaited()
    else:
        mock_pipeline.close.assert_called_once()
        fake_rerank_client.close.assert_called_once()
