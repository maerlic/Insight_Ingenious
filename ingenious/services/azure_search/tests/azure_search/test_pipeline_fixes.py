from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.services.azure_search.components.pipeline import AdvancedSearchPipeline
from ingenious.services.azure_search.config import SearchConfig


@pytest.mark.asyncio
async def test_pipeline_semantic_rerank_id_escaping_or_clause(
    config: SearchConfig, async_iter
):
    """
    Verify that _apply_semantic_ranking constructs a safe OData 'OR' filter
    when document IDs contain commas or single quotes, ensuring the L2 rerank filter is valid.

    NOTE: This test assumes the implementation in pipeline.py has been updated
    to use an 'OR' filter (e.g., id eq 'A,1' or id eq 'B''2') instead of 'search.in()'.
    """
    # Setup: Initialize pipeline with mocked dependencies
    r, f, g = MagicMock(), MagicMock(), MagicMock()
    rerank_client = MagicMock()
    # Configure the rerank client mock; we only care about the arguments it receives.
    # Return an empty iterator as we are focused on the input arguments.
    rerank_client.search = AsyncMock(return_value=async_iter([]))

    # Ensure the config uses a standard 'id' field
    config = config.model_copy(update={"id_field": "id"})

    pipeline = AdvancedSearchPipeline(config, r, f, g, rerank_client=rerank_client)

    # Input: Fused results with IDs containing commas and single quotes
    fused_results = [
        {"id": "A,1", "content": "Doc with comma", "_fused_score": 0.9},
        {"id": "B'2", "content": "Doc with quote", "_fused_score": 0.8},
        {"id": "C", "content": "Normal doc", "_fused_score": 0.7},
    ]

    # Execute
    await pipeline._apply_semantic_ranking("test query", fused_results)

    # Assert: Check the arguments passed to the rerank client's search method
    rerank_client.search.assert_awaited_once()
    call_kwargs = rerank_client.search.call_args.kwargs
    filter_query = call_kwargs.get("filter")

    # The expected filter uses 'eq' and 'or', and escapes the single quote in B'2
    # We check for the presence of each clause rather than the exact string match due to potential ordering differences.
    expected_clauses = [
        "id eq 'A,1'",
        "id eq 'B''2'",  # OData escapes single quotes by doubling them
        "id eq 'C'",
    ]

    assert filter_query is not None
    for clause in expected_clauses:
        assert clause in filter_query, (
            f"Missing or incorrect clause in filter: {clause}\nGenerated filter: {filter_query}"
        )

    # Ensure it doesn't use the vulnerable search.in() syntax
    assert "search.in(" not in filter_query, (
        f"Filter query should use 'OR' clause, not 'search.in()'.\nGenerated filter: {filter_query}"
    )
