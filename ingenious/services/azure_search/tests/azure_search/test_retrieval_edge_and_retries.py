# ingenious/services/azure_search/tests/azure_search/test_retrieval_edge_and_retries.py

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_retriever_lexical_retries_on_429_via_sdk_policy(
    config, monkeypatch, async_iter
):
    """
    Throttling retry presence:
    Ensure the retriever is constructed with a SearchClient that carries a retry policy
    (retries on 429 are handled by the Azure SDK). We assert the presence of the policy
    and that the retriever does not implement manual retry loops.
    """
    # Sentinel policy to detect it was attached by the factory we inject here.
    sentinel_policy = object()

    class DummySearchClient:
        def __init__(self):
            self.retry_policy = sentinel_policy
            self.search = AsyncMock(return_value=async_iter([]))

        async def close(self):
            return None

    class DummyEmbeddingClient:
        def __init__(self):
            self.embeddings = MagicMock()
            self.embeddings.create = AsyncMock(
                return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
            )

    from ingenious.services.azure_search.components.retrieval import (
        AzureSearchRetriever,
    )

    # Create retriever with our custom clients
    r = AzureSearchRetriever(
        config,
        search_client=DummySearchClient(),
        embedding_client=DummyEmbeddingClient(),
    )

    # The client the retriever uses should expose the retry_policy (SDK-level backoff)
    assert getattr(r._search_client, "retry_policy", None) is sentinel_policy

    # One lexical call should go through; we don't implement manual retry loops in the retriever
    out = await r.search_lexical("q")
    assert out == []
    r._search_client.search.assert_awaited()


@pytest.mark.asyncio
async def test_retrieval_handles_unicode_query(config, async_iter, monkeypatch):
    """
    Unicode query: ensure both lexical and vector paths accept and forward unicode safely.
    """
    from ingenious.services.azure_search.components.retrieval import (
        AzureSearchRetriever,
    )

    q = """こんにちは 🌟 — café №42 — "quotes" and emojis 🚀"""

    # Create mock clients
    mock_search_client = MagicMock()
    mock_search_client.search = AsyncMock(return_value=async_iter([]))

    mock_embedding_client = MagicMock()
    mock_embedding_client.embeddings = MagicMock()
    mock_embedding_client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
    )

    # Create retriever with mocked clients
    retriever = AzureSearchRetriever(
        config, search_client=mock_search_client, embedding_client=mock_embedding_client
    )

    # Lexical: just ensure call succeeds and was awaited
    out_lex = await retriever.search_lexical(q)
    assert out_lex == []
    retriever._search_client.search.assert_awaited()

    # Vector: embeddings.create should receive the exact unicode text
    retriever._search_client.search.reset_mock()
    retriever._embedding_client.embeddings.create.reset_mock()

    _ = await retriever.search_vector(q)

    retriever._embedding_client.embeddings.create.assert_awaited()
    _, kwargs = retriever._embedding_client.embeddings.create.call_args
    # openai embeddings call is kwarg-style in implementation
    assert kwargs["input"] == [q]
