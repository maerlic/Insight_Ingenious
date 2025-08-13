# ingenious/services/azure_search/tests/azure_search/test_empty_query_behavior.py

import pytest

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


class _AsyncEmptyResults:
    """Async iterator that yields nothing, mimicking an empty Azure Search page."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _CloseableAioSearchClient:
    """Async SearchClient stub that returns empty results and supports async close()."""

    async def search(self, *args, **kwargs):
        return _AsyncEmptyResults()

    async def close(self):
        return None


class _NoEmbedOpenAI:
    """Embeddings stub that *fails* if called and supports async close()."""

    class _Embeddings:
        async def create(self, *args, **kwargs):
            raise AssertionError(
                "embeddings.create should not be called for empty query"
            )

    def __init__(self):
        self.embeddings = self._Embeddings()

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_retriever_empty_query_returns_empty_or_graceful(config: SearchConfig):
    """
    Both retrieval paths should gracefully return [] for an empty query.
    Vector path should *not* call embeddings for "".
    """
    retriever = AzureSearchRetriever(
        config,
        search_client=_CloseableAioSearchClient(),
        embedding_client=_NoEmbedOpenAI(),
    )

    # 1) Lexical path
    out_lex = await retriever.search_lexical(query="")
    assert out_lex == [], "lexical retrieval should return [] for empty query"

    # 2) Vector path — should short-circuit (and not hit embeddings)
    out_vec = await retriever.search_vector(query="")
    assert out_vec == [], (
        "vector retrieval should return [] and not embed for empty query"
    )

    # Ensure close() is awaitable on both clients
    await retriever.close()


@pytest.mark.asyncio
async def test_pipeline_empty_query_returns_friendly_message(config: SearchConfig):
    """
    End-to-end pipeline should return a friendly, non-empty string and no sources
    when the query is empty, *without* invoking the LLM. Call the real public API.
    """

    class _StubRetriever:
        async def search_lexical(self, _q):
            return []

        async def search_vector(self, _q):
            return []

        async def close(self):
            pass

    class _StubFuser:
        async def fuse(self, _q, _lex, _vec):
            return []

        async def close(self):
            pass

    class _StubAnswerGen:
        async def generate(self, *_a, **_k):
            raise AssertionError("LLM/generation should not be called for empty query")

        async def close(self):
            pass

    pipeline = AdvancedSearchPipeline(
        config,
        _StubRetriever(),
        _StubFuser(),
        _StubAnswerGen(),
    )

    result = await pipeline.get_answer(query="")

    # Tolerate multiple return shapes (tuple, dict, or object with attributes)
    if isinstance(result, tuple):
        answer, sources = result
    elif isinstance(result, dict):
        answer = result.get("answer", "")
        sources = result.get("source_chunks", result.get("sources", []))
    else:
        answer = getattr(result, "answer", "")
        sources = getattr(result, "sources", [])

    assert isinstance(answer, str) and answer.strip(), (
        "friendly message should be a non-empty string"
    )
    assert sources == [], "no sources expected for empty query"
