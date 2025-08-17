"""Test AzureSearchProvider behavior when semantic ranking is disabled.

This module contains tests specifically for the AzureSearchProvider's retrieve
method under the configuration where `use_semantic_ranking` is set to False.
It verifies that the provider correctly falls back to using fused scores,
bypasses the semantic reranking step, and respects the top_k limit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def _make_settings_off_semantic() -> IngeniousSettings:
    """Create a mock IngeniousSettings object with semantic ranking disabled.

    This helper function constructs a settings object tailored for tests that need
    to validate the behavior of the Azure Search provider when semantic ranking
    is turned off in the service configuration.

    Returns:
        An IngeniousSettings instance with semantic ranking disabled.
    """
    s = IngeniousSettings.model_construct()
    s.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="emb",
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
            use_semantic_ranking=False,  # <- OFF
        )
    ]
    return s


@pytest.mark.asyncio
async def test_provider_retrieve_without_semantic_uses_fused_scores_and_top_k(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify retrieve behavior when semantic ranking is off.

    When semantic ranking is disabled via settings, this test ensures that the
    provider.retrieve method:
      - Skips the semantic rerank call.
      - Uses the fused scores from the hybrid search as the final scores.
      - Cleans the internal scoring fields from the final output.
      - Honors the caller's top_k limit on the final result set.
    """
    settings: IngeniousSettings = _make_settings_off_semantic()

    # Mock a pipeline with deterministic fused output
    mock_pipeline: MagicMock = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(return_value=[{"id": "L"}])
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[{"id": "V"}])
    fused: list[dict[str, Any]] = [
        {"id": "A", "content": "A", "_fused_score": 0.9},
        {"id": "B", "content": "B", "_fused_score": 0.8},
        {"id": "C", "content": "C", "_fused_score": 0.7},
    ]
    mock_pipeline.fuser.fuse = AsyncMock(return_value=fused)

    # Rerank client exists but must NOT be used when semantic is off
    fake_rerank_client: MagicMock = MagicMock()
    fake_rerank_client.search = AsyncMock()

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,  # type: ignore[no-untyped-def]
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_rerank_client,  # type: ignore[no-untyped-def]
        raising=False,
    )
    # QueryType shim (not used in OFF-semantic path, but keep consistent with module)
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    provider: AzureSearchProvider = AzureSearchProvider(settings)
    assert provider._cfg.use_semantic_ranking is False

    out: list[dict[str, Any]] = await provider.retrieve("q", top_k=2)

    # Top-K honored
    assert len(out) == 2
    # Reranker was not invoked
    fake_rerank_client.search.assert_not_called()
    # Cleaned outputs: internal fields removed
    doc: dict[str, Any]
    for doc in out:
        k: str
        for k in (
            "_fused_score",
            "_final_score",
            "@search.score",
            "@search.reranker_score",
        ):
            assert k not in doc
