# -- coding: utf-8 --
"""Unit tests for the AzureSearchProvider service.

This module contains a suite of tests for the `AzureSearchProvider`, which is the
primary interface for interacting with the Azure AI Search service. The tests
cover the core public methods: `retrieve()` and `answer()`.

Scenarios tested include:
- The happy path for semantic search and reranking.
- Fallback behavior when semantic reranking cannot be performed.
- Correct delegation of the `answer()` method to the underlying pipeline.
- Graceful handling of both sync and async `close()` methods on dependencies.
- Edge cases like blank queries and disabled answer generation.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider
from ingenious.services.retrieval.errors import GenerationDisabledError

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from pytest import MonkeyPatch


def _make_settings() -> IngeniousSettings:
    """Build a lightweight IngeniousSettings object for provider initialization.

    This helper creates a minimal, valid configuration object required to instantiate
    the AzureSearchProvider without needing a full application configuration load.
    It includes dummy settings for models and Azure Search services.
    """
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
async def test_provider_retrieve_semantic_happy_path(
    monkeypatch: MonkeyPatch,
    async_iter: Callable[[list[dict[str, Any]]], AsyncIterator[dict[str, Any]]],
) -> None:
    """Verify retrieve merges and cleans results on a semantic happy path.

    This test ensures that when a semantic reranker call is successful, its
    results (new scores and updated content) are merged into the fused documents.
    It also confirms that the final output from `retrieve()` is cleaned of all
    internal-only fields like `_final_score` and `@search.reranker_score`.
    """
    settings: IngeniousSettings = _make_settings()

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
    fused: list[dict[str, Any]] = [{"id": "A", "_fused_score": 0.8}]
    internal = await prov._semantic_rerank("q", fused)
    assert internal and internal[0]["_final_score"] == 2.5
    assert internal[0]["content"] == "Alpha"

    # Public: retrieve() should return cleaned version
    out: list[dict[str, Any]] = await prov.retrieve("q", top_k=1)
    assert len(out) == 1
    assert out[0]["id"] == "A"
    assert out[0]["content"] == "Alpha"
    # cleaned fields
    k: str
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
async def test_provider_retrieve_semantic_error_fallback(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify retrieve falls back to fused scores if semantic rerank is skipped.

    This test simulates a scenario where initial search results lack 'id' fields,
    which prevents the semantic reranker from being called. It verifies that the
    provider correctly falls back to using the pre-rerank fused results and
    that the reranker client is never invoked.
    """
    settings: IngeniousSettings = _make_settings()

    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(
        return_value=[{"X": "no-id", "_retrieval_score": 1.0}]
    )
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[])
    fused: list[dict[str, Any]] = [{"X": "no-id", "_fused_score": 0.42, "content": "C"}]
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
    out: list[dict[str, Any]] = await prov.retrieve("q", top_k=1)

    # Should return 1 doc (cleaned), and semantic client wasn't used
    assert len(out) == 1
    doc: dict[str, Any] = out[0]
    k: str
    for k in (
        "_fused_score",
        "_final_score",
        "@search.score",
        "@search.reranker_score",
    ):
        assert k not in doc
    fake_rerank_client.search.assert_not_called()
    await prov.close()


@pytest.mark.parametrize("async_close", [True, False])
@pytest.mark.asyncio
async def test_provider_close_tolerates_sync_or_async(
    monkeypatch: MonkeyPatch,
    async_close: bool,
    async_iter: Callable[[list[dict[str, Any]]], AsyncIterator[dict[str, Any]]],
) -> None:
    """Verify close() handles both synchronous and asynchronous dependencies.

    This test verifies that the `close()` method can correctly await `aclose()`
    on its dependencies if they are async, and call `close()` directly if they
    are sync, without raising an error. This ensures flexibility in the underlying
    client implementations.
    """
    settings: IngeniousSettings = _make_settings()

    # Pipeline with close() sync or async
    mock_pipeline = MagicMock()
    if async_close:
        mock_pipeline.close = AsyncMock()
    else:
        mock_pipeline.close = MagicMock(return_value=None)

    # Rerank client with close() sync or async
    fake_rerank_client = MagicMock()
    fake_rerank_client.search = AsyncMock(return_value=async_iter([]))
    if async_close:
        fake_rerank_client.close = AsyncMock()
    else:
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


@pytest.mark.asyncio
async def test_provider_answer_blank_query_short_circuits(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Verify answer() short-circuits on blank input without calling the pipeline.

    This verifies that providing a blank or whitespace-only query to `answer()`
    does not trigger the search pipeline. Instead, it should return a helpful
    default message and log the event, preventing unnecessary backend calls.
    """
    settings: IngeniousSettings = _make_settings()

    mock_pipeline = MagicMock()
    mock_pipeline.get_answer = AsyncMock()

    fake_rerank_client = MagicMock()

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

    prov = AzureSearchProvider(settings, enable_answer_generation=True)

    # Ensure we capture INFO logs for the provider module's logger
    provider_logger_name: str = AzureSearchProvider.__module__
    with caplog.at_level(logging.INFO, logger=provider_logger_name):
        out: dict[str, Any] = await prov.answer("   \t  ")

    # Assert return payload
    assert out == {
        "answer": "Please enter a question so I can search the knowledge base.",
        "source_chunks": [],
    }

    # Pipeline was not called
    mock_pipeline.get_answer.assert_not_awaited()

    # Assert log message (pick any of these styles)
    assert "Blank query provided; skipping AzureSearchProvider." in caplog.text
    # OR:
    assert any("Blank query provided" in rec.getMessage() for rec in caplog.records)

    await prov.close()


@pytest.mark.asyncio
async def test_provider_answer_raises_when_generation_disabled(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify answer() raises GenerationDisabledError when the feature is off.

    This test ensures that if the provider is configured with answer generation
    disabled, calling the `answer()` method correctly raises a
    `GenerationDisabledError` instead of attempting to contact the pipeline.
    """
    settings: IngeniousSettings = _make_settings()

    mock_pipeline = MagicMock()
    mock_pipeline.get_answer = AsyncMock()

    fake_rerank_client = MagicMock()

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

    # Force-disable via constructor override (takes precedence over settings)
    prov = AzureSearchProvider(settings, enable_answer_generation=False)

    with pytest.raises(GenerationDisabledError) as ei:
        await prov.answer("non-blank")

    # No generation attempted
    mock_pipeline.get_answer.assert_not_awaited()

    # Error details (if your GenerationDisabledError includes these as in the code snippet)
    err: GenerationDisabledError = ei.value
    assert "enable_answer_generation=True" in str(err)
    if hasattr(err, "snapshot"):
        assert isinstance(err.snapshot, dict)
        assert "use_semantic_ranking" in err.snapshot
        assert "top_n_final" in err.snapshot

    await prov.close()


@pytest.mark.asyncio
async def test_provider_answer_passes_query_unmodified(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify answer() passes the raw query string to the pipeline.

    This ensures that the provider does not trim or alter the user's query
    string before passing it to the underlying search pipeline. The only
    preprocessing should be the check for a completely blank query.
    """
    settings: IngeniousSettings = _make_settings()

    captured: dict[str, str] = {}

    async def fake_get_answer(q: str) -> dict[str, Any]:
        """Capture the query and return a dummy answer for verification."""
        captured["q"] = q
        return {"answer": "A", "source_chunks": []}

    mock_pipeline = MagicMock()
    mock_pipeline.get_answer = AsyncMock(side_effect=fake_get_answer)

    fake_rerank_client = MagicMock()

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

    prov = AzureSearchProvider(settings, enable_answer_generation=True)

    messy_query: str = "  some  query   with   spaces  "
    await prov.answer(messy_query)

    assert captured["q"] == messy_query
    await prov.close()
