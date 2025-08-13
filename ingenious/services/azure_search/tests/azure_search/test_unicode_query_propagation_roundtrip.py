from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever


@pytest.mark.asyncio
async def test_unicode_query_roundtrip_lexical_and_vector(config, async_iter):
    """
    Unique coverage:
      - Verifies the *lexical* call passes the exact Unicode query in `search_text`
        (including curly quotes and the numero sign).
      - Ensures embeddings.create sees the same Unicode text.
      - Requires vector path to send `vector_queries` (without duplicating other tests'
        detailed assertions about its internals).
    """
    UNI = "こんにちは 🌟 — café №42 — “quotes” 🚀"

    # --- Search client stub that inspects per-call kwargs --------------------
    call_counter = {"n": 0}

    async def _search_side_effect(*_a, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            # Lexical path: assert the exact Unicode string and top passed through
            assert kwargs.get("search_text") == UNI
            assert kwargs.get("top") == config.top_k_retrieval
        else:
            # Vector path: don't duplicate deep assertions; just ensure vector_queries are present
            assert "vector_queries" in kwargs
        return async_iter([])

    search_client = SimpleNamespace(
        search=AsyncMock(side_effect=_search_side_effect),
        close=AsyncMock(),
    )

    # --- Embeddings client stub that accepts either string or [string] -------
    async def _embed_create(**kwargs):
        got = kwargs.get("input")
        if isinstance(got, list):
            assert len(got) == 1 and got[0] == UNI
        else:
            assert got == UNI
        # Minimal OpenAI-like shape
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 0.0, 0.0])])

    embedding_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(side_effect=_embed_create)),
        close=AsyncMock(),
    )

    # --- Run the retriever under test ----------------------------------------
    retriever = AzureSearchRetriever(
        config, search_client=search_client, embedding_client=embedding_client
    )

    # Lexical: empty iterator → []
    out_lex = await retriever.search_lexical(UNI)
    assert out_lex == []

    # Vector: empty iterator → []
    out_vec = await retriever.search_vector(UNI)
    assert out_vec == []

    # Sanity: both paths executed; embeddings used once
    assert search_client.search.await_count == 2
    assert embedding_client.embeddings.create.await_count == 1

    # Cleanup
    await retriever.close()
    search_client.close.assert_awaited()
    embedding_client.close.assert_awaited()
