"""Test preservation of unmatched documents during semantic reranking.

This module contains tests for the Azure Search service components, specifically
focusing on a critical edge case in the semantic ranking process. It verifies
that when the semantic reranker fails to return a match for a given document
(e.g., due to an ID containing a comma which can break `search.in` filters),
the original document is preserved in the final results rather than being dropped.
This ensures the system is robust against data loss from upstream filtering issues.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline
from ingenious.services.azure_search.provider import AzureSearchProvider

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.asyncio
async def test_apply_semantic_ranking_preserves_unmatched_top50(
    config: Any,
    async_iter: Callable[[list[dict[str, Any]]], AsyncIterator[dict[str, Any]]],
) -> None:
    """Verify `_apply_semantic_ranking` preserves docs if the reranker returns none.

    This test ensures that if the semantic reranker yields no results (e.g.,
    due to an ID with a comma breaking a `search.in()` filter), the original
    fused documents are still present in the final output, preventing data loss.
    """
    # Rerank client that yields NO rows
    rerank_client: MagicMock = MagicMock()
    rerank_client.search = AsyncMock(return_value=async_iter([]))

    # Build a pipeline with dummy components; only _rerank_client is exercised here
    pipeline: AdvancedSearchPipeline = AdvancedSearchPipeline(
        config=config,
        retriever=MagicMock(),
        fuser=MagicMock(),
        answer_generator=MagicMock(),
        rerank_client=rerank_client,
    )

    fused: list[dict[str, Any]] = [
        {"id": "A,1", "content": "alpha", "_fused_score": 0.72},
        {"id": "B", "content": "beta", "_fused_score": 0.55},
    ]

    out: list[dict[str, Any]] = await pipeline._apply_semantic_ranking(
        "any query", fused
    )

    # All top-50 inputs were unmatched; they should be preserved unchanged.
    ids: set[Any] = {d.get("id") for d in out}
    assert {"A,1", "B"} <= ids, (
        f"Expected preservation of unmatched fused docs, got {ids}"
    )

    # The preserved doc retains its fused score/content (no reranker mutation)
    preserved: dict[str, Any] = next(d for d in out if d["id"] == "A,1")
    assert preserved.get("_fused_score") == 0.72
    assert preserved.get("content") == "alpha"


@pytest.mark.asyncio
async def test_provider_semantic_rerank_preserves_unmatched(
    monkeypatch: MonkeyPatch,
    async_iter: Callable[[list[dict[str, Any]]], AsyncIterator[dict[str, Any]]],
) -> None:
    """Verify `_semantic_rerank` preserves docs and sets a fallback score.

    This provider-level test mirrors the pipeline test. It ensures that if the
    reranker returns no matches, the original fused documents are retained and
    their `_final_score` correctly falls back to the `_fused_score`.
    """
    # Construct provider without invoking __init__
    provider: Any = object.__new__(AzureSearchProvider)

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

    fused: list[dict[str, Any]] = [
        {"id": "A,1", "_fused_score": 0.72, "content": "alpha"},
        {"id": "B", "_fused_score": 0.55, "content": "beta"},
    ]

    out: list[dict[str, Any]] = await provider._semantic_rerank("any query", fused)

    # The comma-ID doc must be preserved…
    ids: set[Any] = {d.get("id") for d in out}
    assert "A,1" in ids, f"Expected provider to preserve unmatched fused doc, got {ids}"

    # …and its _final_score should fall back to the fused score.
    doc: dict[str, Any] = next(d for d in out if d["id"] == "A,1")
    assert doc.get("_final_score") == 0.72
    assert doc.get("content") == "alpha"
