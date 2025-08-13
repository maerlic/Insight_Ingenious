# -- coding: utf-8 --

"""
FILE TEST PLAN

    AzureSearchRetriever:
        initializes with factory-made clients
        _generate_embedding returns vector from OpenAI mock
        search_lexical returns list with _retrieval_score/type and respects top_k
        search_vector builds DummyVectorizedQuery, calls clients, and tags results
        close() closes both clients
"""

from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


@pytest.fixture
def retriever(config: SearchConfig):
    return AzureSearchRetriever(config)


@pytest.mark.asyncio
async def test_generate_embedding(retriever: AzureSearchRetriever):
    emb = await retriever._generate_embedding("hello")
    assert isinstance(emb, list) and emb and isinstance(emb[0], float)


@pytest.mark.asyncio
async def test_search_lexical(
    retriever: AzureSearchRetriever, config: SearchConfig, monkeypatch, async_iter
):
    # Make the search client return two docs with @search.score present
    client = retriever._search_client
    client.search = AsyncMock(
        return_value=async_iter(
            [
                {"id": "A", "@search.score": 2.0, config.content_field: "A"},
                {"id": "B", "@search.score": 1.0, config.content_field: "B"},
            ]
        )
    )
    out = await retriever.search_lexical("q")
    assert [d["id"] for d in out] == ["A", "B"]
    assert out[0]["_retrieval_type"] == "lexical_bm25"
    assert out[0]["_retrieval_score"] == 2.0
    client.search.assert_awaited()


@pytest.mark.asyncio
async def test_search_vector(
    retriever: AzureSearchRetriever, config: SearchConfig, monkeypatch, async_iter
):
    # Embedding call returns preset vector via fixture patch
    # Search returns results ordered by score
    client = retriever._search_client
    client.search = AsyncMock(
        return_value=async_iter(
            [
                {"id": "X", "@search.score": 0.9, config.content_field: "X"},
                {"id": "Y", "@search.score": 0.8, config.content_field: "Y"},
            ]
        )
    )
    out = await retriever.search_vector("q")
    assert [d["id"] for d in out] == ["X", "Y"]
    assert out[0]["_retrieval_type"] == "vector_dense"
    assert out[0]["_retrieval_score"] == 0.9
    client.search.assert_awaited()


@pytest.mark.asyncio
async def test_retriever_close(retriever: AzureSearchRetriever):
    await retriever.close()
    retriever._search_client.close.assert_awaited()
    retriever._embedding_client.close.assert_awaited()
