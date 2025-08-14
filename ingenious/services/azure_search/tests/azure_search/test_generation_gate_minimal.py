from typing import Any, Dict, List

import pytest

from ingenious.services.azure_search.components.pipeline import (
    AdvancedSearchPipeline,
    build_search_pipeline,
)

# ──────────────────────────────────────────────────────────────────────────────
# Test stubs
# ──────────────────────────────────────────────────────────────────────────────


class StubRetriever:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.closed = False

    async def search_lexical(self, query: str) -> List[Dict[str, Any]]:
        return [{"id": "L1", "content": "lex-1", "@search.score": 0.2}]

    async def search_vector(self, query: str) -> List[Dict[str, Any]]:
        return [{"id": "V1", "content": "vec-1", "@search.score": 0.3}]

    async def close(self) -> None:
        self.closed = True


class StubFuser:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.closed = False

    async def fuse(
        self,
        query: str,
        lexical_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        docs = [
            {
                "id": "A",
                "content": "A",
                "_fused_score": 0.9,
                "_retrieval_type": "hybrid",
                "vector": [0.1, 0.2],
            },
            {
                "id": "B",
                "content": "B",
                "_fused_score": 0.8,
                "_retrieval_type": "hybrid",
            },
            {
                "id": "C",
                "content": "C",
                "_fused_score": 0.5,
                "_retrieval_type": "hybrid",
            },
            {
                "id": "D",
                "content": "D",
                "_fused_score": 0.1,
                "_retrieval_type": "hybrid",
            },
        ]
        return docs

    async def close(self) -> None:
        self.closed = True


class EmptyFuser(StubFuser):
    async def fuse(self, *_: Any, **__: Any) -> List[Dict[str, Any]]:
        return []


class SpyAnswerGen:
    constructed = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        SpyAnswerGen.constructed += 1
        self.generate_calls = 0
        self.closed = False

    async def generate(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        self.generate_calls += 1
        return "GEN ANSWER"

    async def close(self) -> None:
        self.closed = True


class BoomAnswerGen(SpyAnswerGen):
    async def generate(self, *_: Any, **__: Any) -> str:  # type: ignore[override]
        raise RuntimeError("boom")


class StubRerankClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


# ──────────────────────────────────────────────────────────────────────────────
# build_search_pipeline() construction behavior
# ──────────────────────────────────────────────────────────────────────────────


def test_build_pipeline_generation_disabled_no_generator(monkeypatch, config):
    from ingenious.services.azure_search.components import pipeline as P

    monkeypatch.setattr(P, "AzureSearchRetriever", StubRetriever)
    monkeypatch.setattr(P, "DynamicRankFuser", StubFuser)
    monkeypatch.setattr(P, "AnswerGenerator", SpyAnswerGen)

    cfg = config.copy(update={"enable_answer_generation": False})
    pipe = build_search_pipeline(cfg)

    assert isinstance(pipe, AdvancedSearchPipeline)
    assert pipe.answer_generator is None
    assert SpyAnswerGen.constructed == 0


def test_build_pipeline_generation_enabled_constructs_generator(monkeypatch, config):
    from ingenious.services.azure_search.components import pipeline as P

    SpyAnswerGen.constructed = 0
    monkeypatch.setattr(P, "AzureSearchRetriever", StubRetriever)
    monkeypatch.setattr(P, "DynamicRankFuser", StubFuser)
    monkeypatch.setattr(P, "AnswerGenerator", SpyAnswerGen)

    cfg = config.copy(update={"enable_answer_generation": True})
    pipe = build_search_pipeline(cfg)

    assert pipe.answer_generator is not None
    assert SpyAnswerGen.constructed == 1


# ──────────────────────────────────────────────────────────────────────────────
# get_answer() runtime behavior
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_answer_generation_disabled_returns_retrieval_only(
    monkeypatch, config_no_semantic
):
    cfg = config_no_semantic.copy(
        update={"enable_answer_generation": False, "top_n_final": 3}
    )
    pipe = AdvancedSearchPipeline(
        config=cfg,
        retriever=StubRetriever(cfg),
        fuser=StubFuser(cfg),
        answer_generator=None,  # simulate disabled
        rerank_client=StubRerankClient(),
    )

    res = await pipe.get_answer("q")
    assert res["answer"] == ""  # retrieval-only mode
    assert len(res["source_chunks"]) == 3
    assert all("vector" not in c for c in res["source_chunks"])
    assert [c["id"] for c in res["source_chunks"]] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_get_answer_generation_enabled_calls_generator(
    monkeypatch, config_no_semantic
):
    cfg = config_no_semantic.copy(
        update={"enable_answer_generation": True, "top_n_final": 2}
    )
    gen = SpyAnswerGen(cfg)

    pipe = AdvancedSearchPipeline(
        config=cfg,
        retriever=StubRetriever(cfg),
        fuser=StubFuser(cfg),
        answer_generator=gen,
        rerank_client=StubRerankClient(),
    )

    res = await pipe.get_answer("q")
    assert res["answer"] == "GEN ANSWER"
    assert len(res["source_chunks"]) == 2
    assert gen.generate_calls == 1


@pytest.mark.asyncio
async def test_get_answer_skips_generation_if_flag_false_even_with_generator_object(
    monkeypatch, config_no_semantic
):
    cfg = config_no_semantic.copy(
        update={"enable_answer_generation": False, "top_n_final": 2}
    )
    gen = SpyAnswerGen(cfg)

    pipe = AdvancedSearchPipeline(
        config=cfg,
        retriever=StubRetriever(cfg),
        fuser=StubFuser(cfg),
        answer_generator=gen,  # present, but flag is False
        rerank_client=StubRerankClient(),
    )

    res = await pipe.get_answer("q")
    assert res["answer"] == ""
    assert gen.generate_calls == 0


@pytest.mark.asyncio
async def test_get_answer_no_context_returns_standard_message(
    monkeypatch, config_no_semantic
):
    cfg = config_no_semantic.copy(update={"enable_answer_generation": False})
    pipe = AdvancedSearchPipeline(
        config=cfg,
        retriever=StubRetriever(cfg),
        fuser=EmptyFuser(cfg),
        answer_generator=None,
        rerank_client=StubRerankClient(),
    )

    res = await pipe.get_answer("q")
    assert "could not find any relevant information" in res["answer"].lower()
    assert res["source_chunks"] == []


@pytest.mark.asyncio
async def test_get_answer_generation_error_bubbles_as_runtime_error(
    monkeypatch, config_no_semantic
):
    cfg = config_no_semantic.copy(update={"enable_answer_generation": True})
    gen = BoomAnswerGen(cfg)
    pipe = AdvancedSearchPipeline(
        config=cfg,
        retriever=StubRetriever(cfg),
        fuser=StubFuser(cfg),
        answer_generator=gen,
        rerank_client=StubRerankClient(),
    )

    with pytest.raises(RuntimeError, match="Answer Generation failed"):
        await pipe.get_answer("q")


@pytest.mark.asyncio
async def test_pipeline_close_safely_handles_none_generator(config_no_semantic):
    cfg = config_no_semantic.copy(update={"enable_answer_generation": False})
    retr = StubRetriever(cfg)
    fus = StubFuser(cfg)
    rr = StubRerankClient()

    pipe = AdvancedSearchPipeline(
        config=cfg,
        retriever=retr,
        fuser=fus,
        answer_generator=None,
        rerank_client=rr,
    )

    await pipe.close()
    assert retr.closed is True
    assert fus.closed is True
    assert rr.closed is True
