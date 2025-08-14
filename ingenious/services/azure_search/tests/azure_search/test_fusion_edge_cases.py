# tests/azure_search/test_fusion_edge_cases.py
from unittest.mock import AsyncMock

from ingenious.services.azure_search.components.fusion import DynamicRankFuser


def test_fusion_normalize_scores_degenerate_case_equals_0_5(mock_search_config):
    """
    P3: Verify _normalize_scores sets all scores to 0.5 when max_score == min_score.
    """
    fuser = DynamicRankFuser(config=mock_search_config, llm_client=AsyncMock())

    # Case 1: All scores are the same (non-zero)
    results_same = [
        {"id": "1", "_retrieval_score": 10.0},
        {"id": "2", "_retrieval_score": 10.0},
        {"id": "3", "_retrieval_score": 10.0},
    ]
    fuser._normalize_scores(results_same)
    assert all(r["_normalized_score"] == 0.5 for r in results_same)

    # Case 2: All scores are zero
    results_zero = [
        {"id": "1", "_retrieval_score": 0.0},
        {"id": "2", "_retrieval_score": 0.0},
    ]
    fuser._normalize_scores(results_zero)
    assert all(r["_normalized_score"] == 0.5 for r in results_zero)

    # Case 3: Single result
    results_single = [
        {"id": "1", "_retrieval_score": 5.0},
    ]
    fuser._normalize_scores(results_single)
    assert results_single[0]["_normalized_score"] == 0.5
