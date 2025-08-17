"""Tests the advanced Azure Search pipeline and its factory.

This module contains unit and integration tests for the AdvancedSearchPipeline
and its builder function, `build_search_pipeline`. The tests verify several
key aspects of the search service:
- Correct component wiring by the factory function.
- Validation of configuration settings.
- The logic for applying semantic reranking, including happy paths and fallbacks.
- The cleaning of source documents before they are returned to the user.
- The end-to-end correctness of the `get_answer` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.azure_search.components.fusion import DynamicRankFuser
from ingenious.services.azure_search.components.generation import AnswerGenerator
from ingenious.services.azure_search.components.pipeline import (
    AdvancedSearchPipeline,
    build_search_pipeline,
)
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_build_search_pipeline_success(
    config: SearchConfig, monkeypatch: MonkeyPatch
) -> None:
    """Test the factory constructs the pipeline with correct components."""
    # This test ensures that the `build_search_pipeline` factory function
    # correctly instantiates and wires together the retriever, fuser, and
    # generator components based on the provided configuration.
    MockR = MagicMock()
    MockF = MagicMock()
    MockG = MagicMock()
    monkeypatch.setattr(
        "ingenious.services.azure_search.components.pipeline.AzureSearchRetriever",
        MockR,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.components.pipeline.DynamicRankFuser",
        MockF,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.components.pipeline.AnswerGenerator",
        MockG,
        raising=False,
    )

    # Create a mutable copy and enable answer generation for this test
    mutable_config = config.model_copy(update={"enable_answer_generation": True})

    p = build_search_pipeline(mutable_config)
    assert isinstance(p, AdvancedSearchPipeline)
    MockR.assert_called_once_with(mutable_config)
    MockF.assert_called_once_with(mutable_config)
    MockG.assert_called_once_with(mutable_config)


def test_build_search_pipeline_validation_error(
    config_no_semantic: SearchConfig,
) -> None:
    """Test the factory raises ValueError for invalid semantic ranking config."""
    # This test verifies a key validation rule: if semantic ranking is enabled,
    # the semantic configuration name must be provided. It ensures the factory
    # prevents the creation of a misconfigured pipeline.
    # Since the config model is frozen, we must create a new instance for testing
    invalid_dict = config_no_semantic.model_dump()
    invalid_dict.update(
        {
            "use_semantic_ranking": True,
            "semantic_configuration_name": None,
        }
    )
    invalid = SearchConfig(**invalid_dict)

    with pytest.raises(ValueError):
        build_search_pipeline(invalid)


def test_pipeline_init_sets_rerank_client(config: SearchConfig) -> None:
    """Verify the rerank client is set on pipeline initialization."""
    # The pipeline creates its own reranking client internally, which is a key
    # dependency for semantic ranking. This test confirms that the client is
    # initialized and attached to the instance as expected.
    r, f, g = (
        MagicMock(spec=AzureSearchRetriever),
        MagicMock(spec=DynamicRankFuser),
        MagicMock(spec=AnswerGenerator),
    )
    p = AdvancedSearchPipeline(config, r, f, g)
    assert hasattr(p, "_rerank_client")


@pytest.mark.asyncio
async def test_apply_semantic_ranking_happy(config: SearchConfig) -> None:
    """Test successful semantic reranking applies scores and reorders results."""
    # This test covers the "happy path" for semantic ranking. It ensures that
    # the pipeline correctly calls the reranking API, merges the new scores
    # into the documents, updates content, and reorders them based on the
    # semantic relevance score.
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    p = AdvancedSearchPipeline(config, r, f, g)
    # Build fused results
    fused: list[dict[str, Any]] = [
        {"id": "A", "content": "A", "_fused_score": 0.8, "_retrieval_type": "hybrid"},
        {"id": "B", "content": "B", "_fused_score": 0.7, "_retrieval_type": "vector"},
    ]
    # Rerank returns reversed with scores
    conftest_module: Any = __import__(
        "ingenious.services.azure_search.tests.conftest", fromlist=["AsyncIter"]
    )
    async_iter_mock = conftest_module.AsyncIter(
        [
            {"id": "B", "@search.reranker_score": 3.0, "content": "B2"},
            {"id": "A", "@search.reranker_score": 2.5, "content": "A2"},
        ]
    )

    with patch.object(
        p._rerank_client, "search", AsyncMock(return_value=async_iter_mock)
    ):
        out = await p._apply_semantic_ranking("q", fused)

    assert [d["id"] for d in out] == ["B", "A"]
    assert out[0]["_final_score"] == 3.0
    assert out[0]["_fused_score"] == 0.7
    assert out[0]["content"] == "B2"


@pytest.mark.asyncio
async def test_apply_semantic_ranking_truncation(config: SearchConfig) -> None:
    """Test that input to semantic reranker is truncated to 50 docs."""
    # The Azure Search semantic reranker has a limit of 50 documents per
    # request. This test ensures that the pipeline truncates the input list
    # accordingly, while correctly appending the non-reranked documents to the
    # final result.
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    p = AdvancedSearchPipeline(config, r, f, g)
    fused: list[dict[str, Any]] = [
        {"id": f"doc_{i}", "_fused_score": 1.0} for i in range(55)
    ]

    conftest_module: Any = __import__(
        "ingenious.services.azure_search.tests.conftest", fromlist=["AsyncIter"]
    )
    async_iter_mock = conftest_module.AsyncIter(
        [{"id": f"doc_{i}", "@search.reranker_score": 3.0} for i in range(50)]
    )
    with patch.object(
        p._rerank_client, "search", AsyncMock(return_value=async_iter_mock)
    ):
        out = await p._apply_semantic_ranking("q", fused)

    assert len(out) == 55
    assert out[0]["_final_score"] == 3.0
    assert out[50]["id"] == "doc_50"
    assert "_final_score" not in out[50]


@pytest.mark.asyncio
async def test_apply_semantic_ranking_edge_and_fallback(
    config: SearchConfig,
) -> None:
    """Test semantic ranking fallbacks for empty input, missing IDs, and errors."""
    # This test verifies several failure and edge-case scenarios for semantic
    # ranking: an empty list is handled gracefully, documents missing the
    # required ID field are skipped, and API errors cause a fallback to fused
    # scores, ensuring a result is still returned.
    r, f, g = MagicMock(), MagicMock(), MagicMock()

    # Create a mutable copy for testing different configurations
    mutable_config = config.model_copy(update={})
    p = AdvancedSearchPipeline(mutable_config, r, f, g)

    # Empty input
    assert await p._apply_semantic_ranking("q", []) == []

    # Missing id field: tweak config so id_field not present
    p._config = mutable_config.model_copy(update={"id_field": "nope"})
    fused: list[dict[str, Any]] = [{"id": "A"}]
    assert await p._apply_semantic_ranking("q", fused) == fused

    # API error fallback -> copies _fused_score to _final_score
    p._config = mutable_config.model_copy(update={"id_field": "id"})  # restore
    fused_with_score: list[dict[str, Any]] = [{"id": "A", "_fused_score": 0.9}]

    async def boom(*a: Any, **k: Any) -> None:
        """Simulate a runtime error during an async call."""
        raise RuntimeError("x")

    with patch.object(p._rerank_client, "search", AsyncMock(side_effect=boom)):
        out = await p._apply_semantic_ranking("q", fused_with_score)
    assert out[0]["_final_score"] == 0.9


def test_clean_sources_removes_internal(config: SearchConfig) -> None:
    """Test that internal and verbose metadata fields are cleaned from sources."""
    # To provide a clean and concise API response, the pipeline must remove
    # intermediate scores and Azure-specific metadata. This test confirms that
    # only essential fields (ID, content, final score, retrieval type) are
    # retained.
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    p = AdvancedSearchPipeline(config, r, f, g)
    rows: list[dict[str, Any]] = [
        {
            config.id_field: "1",
            config.content_field: "C",
            "_retrieval_score": 1.0,
            "_normalized_score": 1.0,
            "_fused_score": 0.8,
            "_final_score": 3.3,
            "@search.score": 1.0,
            "@search.reranker_score": 3.3,
            "@search.captions": "cap",
            config.vector_field: [0.1],
            "_retrieval_type": "hybrid",
        }
    ]
    out = p._clean_sources(rows)
    # The id field is now correctly accessed via the config object
    assert out[0][config.id_field] == "1"
    assert out[0]["_final_score"] == 3.3
    assert out[0]["_retrieval_type"] == "hybrid"
    for k in (
        "_retrieval_score",
        "_normalized_score",
        "_fused_score",
        "@search.score",
        "@search.reranker_score",
        "@search.captions",
        cast(str, config.vector_field),
    ):
        assert k not in out[0]


@pytest.mark.asyncio
async def test_get_answer_paths(config: SearchConfig) -> None:
    """Test the end-to-end get_answer method for all major paths."""
    # This is an integration test for the main `get_answer` method. It verifies:
    # 1. The full flow with semantic ranking enabled.
    # 2. The flow without semantic ranking, falling back to fused scores.
    # 3. Correct handling of an empty result set, returning a friendly message.
    # 4. Final source chunks are properly cleaned of verbose metadata.
    config_with_gen = config.model_copy(update={"enable_answer_generation": True})

    # Compose a real pipeline with mocked submethods
    r = MagicMock()
    f = MagicMock()
    g = MagicMock()
    p = AdvancedSearchPipeline(config_with_gen, r, f, g)

    # happy path with semantic
    r.search_lexical = AsyncMock(return_value=[{"id": "L1"}])
    r.search_vector = AsyncMock(return_value=[{"id": "V1"}])
    # FIX 1: The return value now matches the assertion below.
    g.generate = AsyncMock(return_value="final")
    with patch.object(
        p,
        "_apply_semantic_ranking",
        new=AsyncMock(
            return_value=[
                {"id": "S1", "_final_score": 3.0, "content": "C", "vector": [0.1]}
            ]
        ),
    ):
        out = await p.get_answer("q")
    assert out["answer"] == "final"
    assert out["source_chunks"] and "vector" not in out["source_chunks"][0]

    # no results -> friendly message
    # Note: This part re-uses the correctly mocked `r.search_lexical` and `r.search_vector`
    with patch.object(p, "_apply_semantic_ranking", new=AsyncMock(return_value=[])):
        out2 = await p.get_answer("q")
    assert "could not find" in out2["answer"].lower()

    # no semantic path
    cfg2 = config_with_gen.model_copy(update={"use_semantic_ranking": False})
    p2 = AdvancedSearchPipeline(cfg2, r, f, g)
    # FIX 2: All async methods are now mocked directly.
    r.search_lexical = AsyncMock(return_value=[{"id": "L"}])
    r.search_vector = AsyncMock(return_value=[{"id": "V"}])
    f.fuse = AsyncMock(return_value=[{"id": "F", "_fused_score": 0.4}])
    g.generate = AsyncMock(return_value="ans")
    out3 = await p2.get_answer("q")
    assert out3["answer"] == "ans"
    assert out3["source_chunks"][0]["_final_score"] == 0.4  # fused used as final


@pytest.mark.asyncio
async def test_pipeline_close(config: SearchConfig) -> None:
    """Test that the close method propagates to all underlying clients."""
    # Proper resource management is critical. This test ensures that when the
    # pipeline's `close` method is called, it correctly awaits the `close`
    # method of each of its components (retriever, fuser, generator, and the
    # internal reranking client).
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    # add async close methods
    for comp in (r, f, g):
        comp.close = AsyncMock()

    # Create a pipeline instance for the test
    p = AdvancedSearchPipeline(config, r, f, g)
    with patch.object(p._rerank_client, "close", new_callable=AsyncMock) as mock_close:
        await p.close()
        mock_close.assert_awaited()

    r.close.assert_awaited()
    f.close.assert_awaited()
    # g.close() should only be awaited if g is not None
    if g is not None:
        g.close.assert_awaited()
