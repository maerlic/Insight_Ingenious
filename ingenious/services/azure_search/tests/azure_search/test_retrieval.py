# -- coding: utf-8 --
"""Tests for the AzureSearchRetriever component.

This module contains unit tests for the AzureSearchRetriever class, which is
responsible for querying an Azure AI Search index. The tests verify that the
retriever correctly initializes with required clients, generates embeddings,
performs both lexical (BM25) and vector searches, and properly closes its
client connections upon request. Mocks are used extensively to isolate the
retriever from external services like Azure Search and OpenAI during testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable
from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@pytest.fixture
def retriever(config: SearchConfig) -> AzureSearchRetriever:
    """Provides an `AzureSearchRetriever` instance for tests.

    This fixture initializes the retriever with a mock configuration, allowing
    for isolated testing of its methods without actual network calls.
    """
    return AzureSearchRetriever(config)


@pytest.mark.asyncio
async def test_generate_embedding(retriever: AzureSearchRetriever) -> None:
    """Verifies that `_generate_embedding` returns a valid vector.

    This test ensures the internal embedding generation method produces a
    list of floats as expected, confirming integration with the mock
    embedding client.
    """
    emb: list[float] = await retriever._generate_embedding("hello")
    assert isinstance(emb, list) and emb and isinstance(emb[0], float)


@pytest.mark.asyncio
async def test_search_lexical(
    retriever: AzureSearchRetriever,
    config: SearchConfig,
    monkeypatch: MonkeyPatch,
    async_iter: Callable[[list[dict[str, Any]]], AsyncGenerator[dict[str, Any], None]],
) -> None:
    """Tests lexical search, ensuring results are correctly formatted.

    This test mocks the search client's response to a lexical query. It
    verifies that the retriever correctly processes the results, adding the
    `_retrieval_score` and `_retrieval_type` fields, and respects the
    ordering and content of the mocked documents.
    """
    # Make the search client return two docs with @search.score present
    client = retriever._search_client
    mock_search = AsyncMock(
        return_value=async_iter(
            [
                {"id": "A", "@search.score": 2.0, config.content_field: "A"},
                {"id": "B", "@search.score": 1.0, config.content_field: "B"},
            ]
        )
    )
    monkeypatch.setattr(client, "search", mock_search)

    out: list[dict[str, Any]] = await retriever.search_lexical("q")
    assert [d["id"] for d in out] == ["A", "B"]
    assert out[0]["_retrieval_type"] == "lexical_bm25"
    assert out[0]["_retrieval_score"] == 2.0
    mock_search.assert_awaited()


@pytest.mark.asyncio
async def test_search_vector(
    retriever: AzureSearchRetriever,
    config: SearchConfig,
    monkeypatch: MonkeyPatch,
    async_iter: Callable[[list[dict[str, Any]]], AsyncGenerator[dict[str, Any], None]],
) -> None:
    """Tests vector search, ensuring results are correctly formatted.

    This test mocks the embedding and search client responses for a vector
    query. It confirms that the retriever generates a vector query, calls the
    search client, and correctly processes the results by adding metadata
    like `_retrieval_type` and `_retrieval_score`.
    """
    # Embedding call returns preset vector via fixture patch
    # Search returns results ordered by score
    client = retriever._search_client
    mock_search = AsyncMock(
        return_value=async_iter(
            [
                {"id": "X", "@search.score": 0.9, config.content_field: "X"},
                {"id": "Y", "@search.score": 0.8, config.content_field: "Y"},
            ]
        )
    )
    monkeypatch.setattr(client, "search", mock_search)

    out: list[dict[str, Any]] = await retriever.search_vector("q")
    assert [d["id"] for d in out] == ["X", "Y"]
    assert out[0]["_retrieval_type"] == "vector_dense"
    assert out[0]["_retrieval_score"] == 0.9
    mock_search.assert_awaited()


@pytest.mark.asyncio
async def test_retriever_close(retriever: AzureSearchRetriever) -> None:
    """Verifies that the close method properly closes all clients.

    This test ensures that calling `close()` on the retriever correctly
    propagates the close call to both its underlying search client and
    embedding client, preventing resource leaks.
    """
    await retriever.close()
    retriever._search_client.close.assert_awaited()  # type: ignore[attr-defined]
    retriever._embedding_client.close.assert_awaited()  # type: ignore[attr-defined]
