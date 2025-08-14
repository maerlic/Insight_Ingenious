# tests/azure_search/test_pipeline_failures.py

from unittest.mock import AsyncMock

import pytest
from azure.core.exceptions import HttpResponseError

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline


@pytest.mark.asyncio
async def test_pipeline_l2_semantic_ranking_failure_falls_back_to_dat_scores(
    mock_search_config,
):
    """
    P1: Verify _apply_semantic_ranking catches Search Client exceptions and returns the original fused results with _final_score set.
    """
    # Mock components
    mock_retriever = AsyncMock()
    mock_fuser = AsyncMock()
    mock_generator = AsyncMock()
    mock_rerank_client = AsyncMock()

    # Configure the rerank client to fail
    mock_rerank_client.search.side_effect = HttpResponseError(
        "Azure Search Service Unavailable (503)"
    )

    pipeline = AdvancedSearchPipeline(
        config=mock_search_config,
        retriever=mock_retriever,
        fuser=mock_fuser,
        answer_generator=mock_generator,
        rerank_client=mock_rerank_client,
    )

    # Prepare input data (fused results)
    fused_results = [
        {"id": "1", "content": "Doc 1", "_fused_score": 0.9},
        {"id": "2", "content": "Doc 2", "_fused_score": 0.8},
    ]
    query = "test query"

    # Execute the L2 ranking step
    ranked_results = await pipeline._apply_semantic_ranking(query, fused_results)

    # Assert the search client was called
    mock_rerank_client.search.assert_called()

    # Assert the fallback behavior: the results list is the same object (or identical content)
    assert ranked_results == fused_results

    # Assert that _final_score was correctly populated from _fused_score during the fallback
    assert ranked_results[0]["_final_score"] == 0.9
    assert ranked_results[1]["_final_score"] == 0.8
