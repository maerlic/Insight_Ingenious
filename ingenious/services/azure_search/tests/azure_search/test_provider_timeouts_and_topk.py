# tests/azure_search/test_provider_timeouts_and_topk.py
# -*- coding: utf-8 -*-
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider


def _settings(use_semantic=True) -> IngeniousSettings:
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
async def test_provider_retrieve_timeout_propagates(monkeypatch, async_iter):
    """
    If one of the L1 retrieval tasks times out, the exception should propagate.
    """
    settings = _settings()
    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(return_value=[{"id": "L1"}])
    mock_pipeline.retriever.search_vector = AsyncMock(
        side_effect=asyncio.TimeoutError("vec timeout")
    )

    fake_rerank_client = MagicMock()
    fake_rerank_client.search = AsyncMock(return_value=async_iter([]))

    # Wire build seams
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

    prov = AzureSearchProvider(settings)
    with pytest.raises(asyncio.TimeoutError):
        await prov.retrieve("q")

    close = getattr(prov, "close", None)
    if close:
        await close()


@pytest.mark.asyncio
async def test_provider_retrieve_topk_zero_returns_empty(monkeypatch):
    """
    top_k=0 should short-circuit to an empty set and avoid reranking work.
    (We keep semantic_ranking=False to guarantee the reranker path isn't even considered.)
    """
    settings = _settings(use_semantic=False)
    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(return_value=[{"id": "L"}])
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[{"id": "V"}])
    mock_pipeline.fuser.fuse = AsyncMock(
        return_value=[{"id": "A", "_fused_score": 0.9}]
    )

    fake_rerank_client = MagicMock()
    fake_rerank_client.search = AsyncMock()

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

    prov = AzureSearchProvider(settings)
    out = await prov.retrieve("q", top_k=0)
    assert out == []
    fake_rerank_client.search.assert_not_called()

    close = getattr(prov, "close", None)
    if close:
        await close()
