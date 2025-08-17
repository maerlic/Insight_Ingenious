"""Tests AzureSearchProvider timeout and top_k edge cases.

This module provides unit tests for the AzureSearchProvider, focusing on its
resilience and correctness under specific conditions. It verifies that retrieval
timeouts from underlying search tasks are propagated correctly and that a `top_k=0`
request is handled efficiently by short-circuiting, avoiding unnecessary API calls.
These tests use extensive mocking to isolate the provider's logic.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider

if TYPE_CHECKING:
    import pytest


def _settings(use_semantic: bool = True) -> IngeniousSettings:
    """Generate a complete IngeniousSettings object for tests.

    This helper creates a settings instance with predefined model and Azure Search
    configurations, allowing tests to control settings like semantic ranking easily.

    Args:
        use_semantic: Whether to configure the Azure Search service to use
            semantic ranking.

    Returns:
        A fully constructed IngeniousSettings object for testing.
    """
    s = IngeniousSettings.model_construct()
    s.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="emb",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="gpt-4o",
            deployment="chat",
            api_key="K",
            base_url="https://oai",
            api_version="2024-02-01",
        ),
    ]
    s.azure_search_services = [
        AzureSearchSettings(
            service="svc",
            endpoint="https://search.example.net",
            key="SK",
            index_name="idx",
            use_semantic_ranking=use_semantic,
        ),
    ]
    return s


@pytest.mark.asyncio
async def test_provider_retrieve_timeout_propagates(
    monkeypatch: pytest.MonkeyPatch, async_iter: Any
) -> None:
    """Test that an L1 retrieval task timeout propagates correctly.

    This test ensures that if one of the underlying search tasks (lexical or
    vector) raises an `asyncio.TimeoutError`, the main `retrieve` method
    also raises it instead of suppressing it.
    """
    settings: IngeniousSettings = _settings()
    mock_pipeline: MagicMock = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(return_value=[{"id": "L1"}])
    mock_pipeline.retriever.search_vector = AsyncMock(
        side_effect=asyncio.TimeoutError("vec timeout")
    )

    fake_rerank_client: MagicMock = MagicMock()
    fake_rerank_client.search = AsyncMock(return_value=async_iter([]))

    def _mock_build_pipeline(_: AzureSearchSettings) -> MagicMock:
        """Return the closure's mocked pipeline instance."""
        return mock_pipeline

    def _mock_make_client(_: AzureSearchSettings) -> MagicMock:
        """Return the closure's mocked search client instance."""
        return fake_rerank_client

    # Wire build seams
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        _mock_build_pipeline,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        _mock_make_client,
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    with pytest.raises(asyncio.TimeoutError):
        await prov.retrieve("q")

    close: Callable[[], Awaitable[None]] | None = getattr(prov, "close", None)
    if close:
        await close()


@pytest.mark.asyncio
async def test_provider_retrieve_topk_zero_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that `top_k=0` short-circuits to return an empty list.

    This test verifies that a `top_k` value of zero efficiently returns an empty
    list without performing any search or reranking operations. This is an
    important optimization. We set `use_semantic=False` to guarantee the
    reranker path is not considered.
    """
    settings: IngeniousSettings = _settings(use_semantic=False)
    mock_pipeline: MagicMock = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(return_value=[{"id": "L"}])
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[{"id": "V"}])
    mock_pipeline.fuser.fuse = AsyncMock(
        return_value=[{"id": "A", "_fused_score": 0.9}]
    )

    fake_rerank_client: MagicMock = MagicMock()
    fake_rerank_client.search = AsyncMock()

    def _mock_build_pipeline(_: AzureSearchSettings) -> MagicMock:
        """Return the closure's mocked pipeline instance."""
        return mock_pipeline

    def _mock_make_client(_: AzureSearchSettings) -> MagicMock:
        """Return the closure's mocked search client instance."""
        return fake_rerank_client

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        _mock_build_pipeline,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        _mock_make_client,
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    out: list[dict[str, Any]] = await prov.retrieve("q", top_k=0)
    assert out == []
    fake_rerank_client.search.assert_not_called()

    close: Callable[[], Awaitable[None]] | None = getattr(prov, "close", None)
    if close:
        await close()
