"""
Tests graceful handling of empty queries in the Azure Search service.

This module verifies that the core components, specifically the AzureSearchRetriever
and the AdvancedSearchPipeline, behave correctly and efficiently when given an
empty or whitespace-only query string. The primary goal is to prevent unnecessary
and potentially costly downstream API calls, such as to OpenAI for embeddings or
to a large language model for answer generation. Instead, the system should return
a graceful, empty result or a helpful message.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, NoReturn

import pytest

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


class _AsyncEmptyResults:
    """Async iterator that yields nothing, mimicking an empty Azure Search page."""

    def __aiter__(self) -> AsyncIterator[Any]:
        """Return the async iterator object."""
        return self

    async def __anext__(self) -> NoReturn:
        """Raise StopAsyncIteration to signal the end of an empty sequence."""
        raise StopAsyncIteration


class _CloseableAioSearchClient:
    """Async SearchClient stub that returns empty results and supports async close()."""

    async def search(self, *args: Any, **kwargs: Any) -> _AsyncEmptyResults:
        """Mimic the search method to always return an empty async iterator."""
        return _AsyncEmptyResults()

    async def close(self) -> None:
        """Provide a no-op, awaitable close method for compatibility."""
        return None


class _NoEmbedOpenAI:
    """Embeddings stub that *fails* if called and supports async close()."""

    class _Embeddings:
        """A stub for the 'embeddings' attribute of the OpenAI client."""

        async def create(self, *args: Any, **kwargs: Any) -> NoReturn:
            """
            Fail the test if this method is ever called.

            This assertion is the core of the test, ensuring that no attempt is made
            to generate embeddings for an empty string, which is an invalid and
            wasteful operation.
            """
            raise AssertionError(
                "embeddings.create should not be called for empty query"
            )

    def __init__(self) -> None:
        """Initialize the stub with a failing embeddings creator."""
        self.embeddings: _NoEmbedOpenAI._Embeddings = self._Embeddings()

    async def close(self) -> None:
        """Provide a no-op, awaitable close method for compatibility."""
        return None


@pytest.mark.asyncio
async def test_retriever_empty_query_returns_empty_or_graceful(
    config: SearchConfig,
) -> None:
    """
    Verify the retriever gracefully returns an empty list for an empty query.

    This test ensures that both the lexical and vector search paths handle an empty
    query string by returning an empty list of results. Crucially, it confirms
    that the vector path short-circuits and does not attempt to call the
    embeddings service, which would be an error.
    """
    retriever = AzureSearchRetriever(
        config,
        search_client=_CloseableAioSearchClient(),
        embedding_client=_NoEmbedOpenAI(),
    )

    # 1) Lexical path
    out_lex: list[Any] = await retriever.search_lexical(query="")
    assert out_lex == [], "lexical retrieval should return [] for empty query"

    # 2) Vector path — should short-circuit (and not hit embeddings)
    out_vec: list[Any] = await retriever.search_vector(query="")
    assert out_vec == [], (
        "vector retrieval should return [] and not embed for empty query"
    )

    # Ensure close() is awaitable on both clients
    await retriever.close()


@pytest.mark.asyncio
async def test_pipeline_empty_query_returns_friendly_message(
    config: SearchConfig,
) -> None:
    """
    Verify the pipeline returns a friendly message for an empty query without calling the LLM.

    This end-to-end test checks that the public `get_answer` method of the
    `AdvancedSearchPipeline` bypasses the entire retrieval, fusion, and generation
    process when the query is empty. It should instead return a helpful, non-empty
    string and no source documents, preventing unnecessary API calls.
    """

    class _StubRetriever:
        """A retriever stub that always returns empty search results."""

        async def search_lexical(self, _q: str) -> list[Any]:
            """Simulate lexical search returning no results."""
            return []

        async def search_vector(self, _q: str) -> list[Any]:
            """Simulate vector search returning no results."""
            return []

        async def close(self) -> None:
            """Provide a no-op, awaitable close method."""
            pass

    class _StubFuser:
        """A fuser stub that always returns an empty list of fused results."""

        async def fuse(self, _q: str, _lex: list[Any], _vec: list[Any]) -> list[Any]:
            """Simulate result fusion, always returning an empty list."""
            return []

        async def close(self) -> None:
            """Provide a no-op, awaitable close method."""
            pass

    class _StubAnswerGen:
        """An answer generator stub that fails if its generate method is called."""

        async def generate(self, *args: Any, **kwargs: Any) -> NoReturn:
            """
            Fail the test if this method is ever called.

            This assertion confirms that the pipeline short-circuits before
            invoking the LLM for answer generation when the query is empty.
            """
            raise AssertionError("LLM/generation should not be called for empty query")

        async def close(self) -> None:
            """Provide a no-op, awaitable close method."""
            pass

    config = config.copy(update={"enable_answer_generation": True})
    pipeline = AdvancedSearchPipeline(
        config,
        _StubRetriever(),
        _StubFuser(),
        _StubAnswerGen(),
    )

    result: Any = await pipeline.get_answer(query="")

    # Tolerate multiple return shapes (tuple, dict, or object with attributes)
    answer: str = ""
    sources: list[Any] = []
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
