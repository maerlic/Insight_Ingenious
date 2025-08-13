# tests/services/azure_search/test_semantic_rerank_edge_filter.py

from unittest.mock import AsyncMock

import pytest

# ⬇️ Adjust this import to your project layout if needed
PipelineMod = pytest.importorskip("ingenious.services.azure_search.pipeline")
AdvancedSearchPipeline = PipelineMod.AdvancedSearchPipeline


@pytest.mark.asyncio
async def test_semantic_rerank_preserves_docs_when_id_has_comma(monkeypatch):
    """
    When a fused doc's ID contains commas (which collide with search.in() delimiters),
    the pipeline should still preserve that doc in the final results even if the
    semantic rerank call returns *no* matches (e.g., because those IDs were excluded
    from the filter). This turns the previous xfail into a pass by asserting that
    the "unsafe-ID" fused doc is appended unchanged.
    """
    # Make a pipeline instance without running its __init__
    pipeline = AdvancedSearchPipeline.__new__(AdvancedSearchPipeline)

    # Force the internal semantic rerank call to yield no results
    # (simulating the "filter ate my comma-ID" scenario).
    monkeypatch.setattr(pipeline, "_semantic_rerank", AsyncMock(return_value=[]))

    # Minimal knobs the method might look at; harmless if unused.
    # Set a generous cap so test data isn't truncated.
    setattr(pipeline, "_MAX_RERANK", 50)

    fused = [
        {
            "id": "doc,1",  # <- the problematic ID with commas
            "content": "alpha",
            "_fused_score": 0.72,
        },
        {
            "id": "doc-2",
            "content": "beta",
            "_fused_score": 0.55,
        },
    ]

    # Some codebases call this helper directly; others call it inside get_answer().
    # We invoke it directly to focus on the merge behavior.
    # If your signature differs (e.g., no top_n), remove that argument below.
    final = await pipeline._apply_semantic_ranking(  # type: ignore[attr-defined]
        query="any query",
        fused_results=fused,
        top_n_final=2,
    )

    # The doc with a comma in its ID must still be present.
    final_ids = [d.get("id") for d in final]
    assert "doc,1" in final_ids, "fused doc with comma in ID should be preserved"

    # And it should be the unchanged fused doc (score kept, not mutated)
    preserved = next(d for d in final if d["id"] == "doc,1")
    assert preserved.get("_fused_score") == 0.72
