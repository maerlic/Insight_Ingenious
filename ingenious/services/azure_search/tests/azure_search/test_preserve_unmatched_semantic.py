# ingenious/services/azure_search/tests/azure_search/test_preserve_unmatched_semantic.py
# -- coding: utf-8 -*-

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline
from ingenious.services.azure_search.provider import AzureSearchProvider


@pytest.mark.asyncio
async def test_apply_semantic_ranking_preserves_unmatched_top50(config, async_iter):
    """
    When the reranker returns no rows (e.g., because an ID contains a comma and search.in()
    split it), the fused doc(s) must still be present in the final output.
    """
    # Rerank client that yields NO rows
    rerank_client = MagicMock()
    rerank_client.search = AsyncMock(return_value=async_iter([]))

    # Build a pipeline with dummy components; only _rerank_client is exercised here
    pipeline = AdvancedSearchPipeline(
        config=config,
        retriever=MagicMock(),
        fuser=MagicMock(),
        answer_generator=MagicMock(),
        rerank_client=rerank_client,
    )

    fused = [
        {"id": "A,1", "content": "alpha", "_fused_score": 0.72},
        {"id": "B", "content": "beta", "_fused_score": 0.55},
    ]

    out = await pipeline._apply_semantic_ranking("any query", fused)

    # All top-50 inputs were unmatched; they should be preserved unchanged.
    ids = {d.get("id") for d in out}
    assert {"A,1", "B"} <= ids, (
        f"Expected preservation of unmatched fused docs, got {ids}"
    )

    # The preserved doc retains its fused score/content (no reranker mutation)
    preserved = next(d for d in out if d["id"] == "A,1")
    assert preserved.get("_fused_score") == 0.72
    assert preserved.get("content") == "alpha"


@pytest.mark.asyncio
async def test_provider_semantic_rerank_preserves_unmatched(monkeypatch, async_iter):
    """
    Provider-level mirror: if the semantic rerank returns no rows, the fused doc must be
    retained and _final_score should fall back to _fused_score.
    """
    # Construct provider without invoking __init__
    provider = object.__new__(AzureSearchProvider)

    # Minimal cfg the helper consults
    provider._cfg = SimpleNamespace(
        id_field="id",
        semantic_configuration_name="default",
    )

    # Patch QueryType on the provider module (keeps us offline)
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    # Rerank client yields NO rows
    provider._rerank_client = MagicMock()
    provider._rerank_client.search = AsyncMock(return_value=async_iter([]))

    fused = [
        {"id": "A,1", "_fused_score": 0.72, "content": "alpha"},
        {"id": "B", "_fused_score": 0.55, "content": "beta"},
    ]

    out = await provider._semantic_rerank("any query", fused)

    # The comma-ID doc must be preserved…
    ids = {d.get("id") for d in out}
    assert "A,1" in ids, f"Expected provider to preserve unmatched fused doc, got {ids}"

    # …and its _final_score should fall back to the fused score.
    doc = next(d for d in out if d["id"] == "A,1")
    assert doc.get("_final_score") == 0.72
    assert doc.get("content") == "alpha"
