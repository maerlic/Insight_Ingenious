"""
FILE TEST PLAN

    build_search_pipeline:
        successful wiring and validation error when semantic ranking required but name missing

    AdvancedSearchPipeline:
        init sets rerank client via factory
        _apply_semantic_ranking happy path (filter construction, merging, ordering, fallback append)
        _apply_semantic_ranking truncates to 50 docs
        _apply_semantic_ranking falls back when errors/missing IDs/empty
        _clean_sources removes internal & azure metadata but keeps essentials
        get_answer end-to-end: with semantic ON and OFF paths; empty top-N returns friendly message
        close() closes all clients
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.services.azure_search.components.fusion import DynamicRankFuser
from ingenious.services.azure_search.components.generation import AnswerGenerator
from ingenious.services.azure_search.components.pipeline import (
    AdvancedSearchPipeline,
    build_search_pipeline,
)
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


def test_build_search_pipeline_success(config: SearchConfig, monkeypatch):
    # Patch components to identifiable sentinels
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


def test_build_search_pipeline_validation_error(config_no_semantic: SearchConfig):
    # Flip to invalid state
    data = config_no_semantic.model_dump(exclude={"search_key", "openai_key"})
    data["use_semantic_ranking"] = True
    data["semantic_configuration_name"] = None
    data["search_key"] = config_no_semantic.search_key
    data["openai_key"] = config_no_semantic.openai_key

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


def test_pipeline_init_sets_rerank_client(config: SearchConfig):
    r, f, g = (
        MagicMock(spec=AzureSearchRetriever),
        MagicMock(spec=DynamicRankFuser),
        MagicMock(spec=AnswerGenerator),
    )
    p = AdvancedSearchPipeline(config, r, f, g)
    assert hasattr(p, "_rerank_client")


@pytest.mark.asyncio
async def test_apply_semantic_ranking_happy(config: SearchConfig, monkeypatch):
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    p = AdvancedSearchPipeline(config, r, f, g)
    # Build fused results
    fused = [
        {"id": "A", "content": "A", "_fused_score": 0.8, "_retrieval_type": "hybrid"},
        {"id": "B", "content": "B", "_fused_score": 0.7, "_retrieval_type": "vector"},
    ]
    # Rerank returns reversed with scores
    async_iter_mock = __import__(
        "ingenious.services.azure_search.tests.conftest", fromlist=["AsyncIter"]
    ).AsyncIter(
        [
            {"id": "B", "@search.reranker_score": 3.0, "content": "B2"},
            {"id": "A", "@search.reranker_score": 2.5, "content": "A2"},
        ]
    )
    p._rerank_client.search = AsyncMock(return_value=async_iter_mock)

    out = await p._apply_semantic_ranking("q", fused)
    assert [d["id"] for d in out] == ["B", "A"]
    assert out[0]["_final_score"] == 3.0
    assert out[0]["_fused_score"] == 0.7
    assert out[0]["content"] == "B2"


@pytest.mark.asyncio
async def test_apply_semantic_ranking_truncation(config: SearchConfig):
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    p = AdvancedSearchPipeline(config, r, f, g)
    fused = [{"id": f"doc_{i}", "_fused_score": 1.0} for i in range(55)]

    async_iter_mock = __import__(
        "ingenious.services.azure_search.tests.conftest", fromlist=["AsyncIter"]
    ).AsyncIter([{"id": f"doc_{i}", "@search.reranker_score": 3.0} for i in range(50)])
    p._rerank_client.search = AsyncMock(return_value=async_iter_mock)

    out = await p._apply_semantic_ranking("q", fused)
    assert len(out) == 55
    assert out[0]["_final_score"] == 3.0
    assert out[50]["id"] == "doc_50"
    assert "_final_score" not in out[50]


@pytest.mark.asyncio
async def test_apply_semantic_ranking_edge_and_fallback(config: SearchConfig):
    r, f, g = MagicMock(), MagicMock(), MagicMock()

    # Create a mutable copy for testing different configurations
    mutable_config = config.model_copy(update={})
    p = AdvancedSearchPipeline(mutable_config, r, f, g)

    # Empty input
    assert await p._apply_semantic_ranking("q", []) == []

    # Missing id field: tweak config so id_field not present
    p._config = mutable_config.model_copy(update={"id_field": "nope"})
    fused = [{"id": "A"}]
    assert await p._apply_semantic_ranking("q", fused) == fused

    # API error fallback -> copies _fused_score to _final_score
    p._config = mutable_config.model_copy(update={"id_field": "id"})  # restore
    fused = [{"id": "A", "_fused_score": 0.9}]

    async def boom(*a, **k):
        raise RuntimeError("x")

    p._rerank_client.search = AsyncMock(side_effect=boom)
    out = await p._apply_semantic_ranking("q", fused)
    assert out[0]["_final_score"] == 0.9


def test_clean_sources_removes_internal(config: SearchConfig):
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    p = AdvancedSearchPipeline(config, r, f, g)
    rows = [
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
        config.vector_field,
    ):
        assert k not in out[0]


@pytest.mark.asyncio
async def test_get_answer_paths(config: SearchConfig, monkeypatch):
    # Create a mutable copy and enable answer generation for the happy path
    config_with_gen = config.model_copy(update={"enable_answer_generation": True})

    # Compose a real pipeline with mocked submethods
    r = MagicMock()
    f = MagicMock()
    g = MagicMock()
    p = AdvancedSearchPipeline(config_with_gen, r, f, g)

    # happy path with semantic
    r.search_lexical = AsyncMock(return_value=[{"id": "L1"}])
    r.search_vector = AsyncMock(return_value=[{"id": "V1"}])
    p._apply_semantic_ranking = AsyncMock(
        return_value=[
            {"id": "S1", "_final_score": 3.0, "content": "C", "vector": [0.1]}
        ]
    )
    g.generate = AsyncMock(return_value="final")
    out = await p.get_answer("q")
    assert out["answer"] == "final"
    assert out["source_chunks"] and "vector" not in out["source_chunks"][0]

    # no results -> friendly message
    p._apply_semantic_ranking = AsyncMock(return_value=[])
    out2 = await p.get_answer("q")
    assert "could not find" in out2["answer"].lower()

    # no semantic path
    cfg2 = config_with_gen.model_copy(update={"use_semantic_ranking": False})
    p2 = AdvancedSearchPipeline(cfg2, r, f, g)
    r.search_lexical = AsyncMock(return_value=[{"id": "L"}])
    r.search_vector = AsyncMock(return_value=[{"id": "V"}])
    f.fuse = AsyncMock(return_value=[{"id": "F", "_fused_score": 0.4}])
    g.generate = AsyncMock(return_value="ans")
    out3 = await p2.get_answer("q")
    assert out3["answer"] == "ans"
    assert out3["source_chunks"][0]["_final_score"] == 0.4  # fused used as final


@pytest.mark.asyncio
async def test_pipeline_close(config: SearchConfig):
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    # add async close methods
    for comp in (r, f, g):
        comp.close = AsyncMock()

    # Create a pipeline instance for the test
    p = AdvancedSearchPipeline(config, r, f, g)
    p._rerank_client.close = AsyncMock()
    await p.close()
    r.close.assert_awaited()
    f.close.assert_awaited()
    # g.close() should only be awaited if g is not None
    if g is not None:
        g.close.assert_awaited()
    p._rerank_client.close.assert_awaited()
