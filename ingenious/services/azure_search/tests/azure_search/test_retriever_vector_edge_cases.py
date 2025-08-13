# ingenious/services/azure_search/tests/azure_search/test_retriever_vector_edge_cases.py
# -- coding: utf-8 --

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


@pytest.mark.asyncio
async def test_retriever_empty_query_short_circuits_embedding(
    config: SearchConfig, async_iter
):
    """
    Purpose:
        Ensure vector path does not call embeddings (or, at minimum, returns [])
        for query == "".

    Uses:
        - `config` and `async_iter` fixtures from conftest.py
        - Stubs embedding client `.embeddings.create` and asserts NOT called
        - Stubs SearchClient.search to return empty async iterator and asserts NOT called
    """
    # Embedding client: must NOT be called for empty query
    embedding_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock()),
        close=AsyncMock(),
    )

    # Search client: should also NOT be called if short-circuiting is in place
    search_client = SimpleNamespace(
        search=AsyncMock(return_value=async_iter([])),
        close=AsyncMock(),
    )

    retriever = AzureSearchRetriever(
        config=config,
        search_client=search_client,
        embedding_client=embedding_client,
    )

    out = await retriever.search_vector("")  # empty query

    # Expect graceful empty result and no downstream calls
    assert out == [], "Vector path should return [] for empty query"
    embedding_client.embeddings.create.assert_not_called()
    search_client.search.assert_not_called()

    # Clean-up
    await retriever.close()
    search_client.close.assert_awaited()
    embedding_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_retriever_unicode_query_vector_path(config: SearchConfig, async_iter):
    """
    Purpose:
        Exercise vector path with unicode; ensure embeddings called once and
        search invoked with a VectorizedQuery-equivalent (as patched in conftest.py).

    Uses:
        - `config` from conftest.py (top_k_retrieval/id/content/vector fields)
        - `async_iter` from conftest.py
        - conftest's DummyVectorizedQuery via patched module attribute
    """
    query = "naïve café ☕️ – 東京"

    # Embeddings.create returns a fixed vector
    embedding_vector = [0.01, 0.02, 0.03]
    embeddings_create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=embedding_vector)])
    )
    embedding_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=embeddings_create),
        close=AsyncMock(),
    )

    # Search client yields unicode docs
    docs = [
        {"id": "γ", "@search.score": 0.93, config.content_field: "smörgåsbord"},
        {"id": "δ", "@search.score": 0.81, config.content_field: "renée"},
    ]
    search_mock = AsyncMock(return_value=async_iter(docs))
    search_client = SimpleNamespace(search=search_mock, close=AsyncMock())

    retriever = AzureSearchRetriever(
        config=config,
        search_client=search_client,
        embedding_client=embedding_client,
    )

    out = await retriever.search_vector(query)

    # Output order and tagging
    assert [d["id"] for d in out] == ["γ", "δ"]
    assert out[0]["_retrieval_type"] == "vector_dense"
    assert out[0]["_retrieval_score"] == 0.93

    # Embeddings were invoked once with expected params
    embeddings_create.assert_awaited_once()
    e_kwargs = embeddings_create.call_args.kwargs
    assert e_kwargs.get("input") == [query]
    assert e_kwargs.get("model") == config.embedding_deployment_name

    # Search invoked with a single DummyVectorizedQuery (patched in conftest.py)
    search_mock.assert_awaited_once()
    s_kwargs = search_mock.call_args.kwargs
    assert s_kwargs.get("search_text") is None
    assert s_kwargs.get("top") == config.top_k_retrieval

    vqs = s_kwargs.get("vector_queries")
    assert isinstance(vqs, list) and len(vqs) == 1
    vq = vqs[0]
    # Conftest patches VectorizedQuery → DummyVectorizedQuery; introspect attrs
    assert getattr(vq, "vector", None) == embedding_vector
    assert getattr(vq, "k_nearest_neighbors", None) == config.top_k_retrieval
    assert getattr(vq, "fields", None) == config.vector_field
    assert getattr(vq, "exhaustive", None) is True

    # Clean-up
    await retriever.close()
    search_client.close.assert_awaited()
    embedding_client.close.assert_awaited()
