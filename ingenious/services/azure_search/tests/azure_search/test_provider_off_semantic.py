# ingenious/services/azure_search/tests/azure_search/test_provider_off_semantic.py

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider


def _make_settings_off_semantic() -> IngeniousSettings:
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
    monkeypatch,
):
    """
    Provider OFF-semantic/top_k:
    When semantic is disabled via settings, provider.retrieve should:
      - skip semantic rerank (no reranker call),
      - use fused scores as final (then clean them out for caller),
      - honor the caller's top_k limit.
    """
    settings = _make_settings_off_semantic()

    # Mock a pipeline with deterministic fused output
    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(return_value=[{"id": "L"}])
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[{"id": "V"}])
    fused = [
        {"id": "A", "content": "A", "_fused_score": 0.9},
        {"id": "B", "content": "B", "_fused_score": 0.8},
        {"id": "C", "content": "C", "_fused_score": 0.7},
    ]
    mock_pipeline.fuser.fuse = AsyncMock(return_value=fused)

    # Rerank client exists but must NOT be used when semantic is off
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
    # QueryType shim (not used in OFF-semantic path, but keep consistent with module)
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    provider = AzureSearchProvider(settings)
    assert provider._cfg.use_semantic_ranking is False

    out = await provider.retrieve("q", top_k=2)

    # Top-K honored
    assert len(out) == 2
    # Reranker was not invoked
    fake_rerank_client.search.assert_not_called()
    # Cleaned outputs: internal fields removed
    for doc in out:
        for k in (
            "_fused_score",
            "_final_score",
            "@search.score",
            "@search.reranker_score",
        ):
            assert k not in doc
