# ingenious/services/azure_search/tests/azure_search/test_unicode_and_commas.py
# -*- coding: utf-8 -*-
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


class _CloseableAioSearchClient:
    """Async SearchClient stub with overrideable search() and async close()."""

    def __init__(self):
        # default to empty iterator; tests will override .search with AsyncMock
        self.search = AsyncMock(side_effect=lambda *a, **k: _AsyncEmptyResults())

    async def close(self):
        return None


class _AsyncEmptyResults:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _EmbeddingClientWithVector:
    """Embedding client that returns a fixed embedding and supports async close()."""

    class _Embeddings:
        def __init__(self, vector):
            self._vector = list(vector)

        async def create(self, *args, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=self._vector)])

    def __init__(self, vector):
        self.embeddings = self._Embeddings(vector)

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_retriever_unicode_query_ok(config: SearchConfig, async_iter):
    """
    Ensure Unicode/special chars in the query do not break lexical or vector paths.
    """
    search_client = _CloseableAioSearchClient()
    embedding_client = _EmbeddingClientWithVector([0.01, 0.02, 0.03])

    retriever = AzureSearchRetriever(
        config,
        search_client=search_client,
        embedding_client=embedding_client,
    )

    # Lexical path: return two rows
    search_client.search = AsyncMock(
        return_value=async_iter(
            [
                {
                    "id": "α",
                    "@search.score": 2.1,
                    config.content_field: "naïve café ☕️ – 東京",
                },
                {"id": "β", "@search.score": 1.4, config.content_field: "crème brûlée"},
            ]
        )
    )
    out_lex = await retriever.search_lexical("naïve café ☕️ – 東京")
    assert [d["id"] for d in out_lex] == ["α", "β"]

    # Vector path: embedding comes from our stub's embeddings.create(...)
    search_client.search = AsyncMock(
        return_value=async_iter(
            [
                {"id": "γ", "@search.score": 0.93, config.content_field: "smörgåsbord"},
                {"id": "δ", "@search.score": 0.81, config.content_field: "renée"},
            ]
        )
    )
    out_vec = await retriever.search_vector("naïve café ☕️ – 東京")
    assert [d["id"] for d in out_vec] == ["γ", "δ"]

    # Ensure async close() exists and is awaitable on both clients
    await retriever.close()


@pytest.mark.xfail(
    reason="IDs with commas break search.in(...) filter; current code may drop such docs. Consider escaping or fallback.",
    strict=False,
)
@pytest.mark.asyncio
async def test_apply_semantic_ranking_ids_with_commas_xfail(
    config: SearchConfig, async_iter
):
    """
    If a fused result ID contains a comma, the rerank filter can become ambiguous.
    Desired behavior: gracefully keep the doc (fallback to fused) even if reranker returns nothing.
    """
    pipeline = AdvancedSearchPipeline(config, MagicMock(), MagicMock(), MagicMock())

    # If the private helper doesn't exist in your version, skip this xfail.
    if not hasattr(pipeline, "_apply_semantic_ranking"):
        pytest.skip(
            "AdvancedSearchPipeline._apply_semantic_ranking() not present; API changed."
        )

    # Make sure the reranker client returns nothing (what happens with malformed filters)
    pipeline._rerank_client = MagicMock()
    pipeline._rerank_client.search = AsyncMock(return_value=async_iter([]))

    fused = [{"id": "A,1", "_fused_score": 0.7, config.content_field: "C"}]

    maybe = pipeline._apply_semantic_ranking("Q", fused)
    out = await maybe if inspect.isawaitable(maybe) else maybe

    # Desired contract (xfail): keep the doc by falling back to fused scores
    assert len(out) == 1 and out[0]["id"] == "A,1"
