# -*- coding: utf-8 -*-

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.services.azure_search.provider import AzureSearchProvider


@pytest.mark.asyncio
async def test_provider_semantic_filter_escaping_or_clause(monkeypatch, async_iter):
    """
    Provider-level check: _semantic_rerank must construct a safe OData 'OR' filter
    for IDs containing commas and single quotes, and must NOT use search.in(...).
    """
    # Build provider without calling __init__
    provider = object.__new__(AzureSearchProvider)

    # Minimal config needed by _semantic_rerank
    provider._cfg = SimpleNamespace(
        id_field="id",
        semantic_configuration_name="default",
    )

    # Ensure QueryType is available in the provider module
    # Use the DummyQueryType from conftest
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
        raising=False,
    )

    # Rerank client: we only care about the arguments it receives
    provider._rerank_client = MagicMock()
    # Use the AsyncIter class from conftest via the fixture
    provider._rerank_client.search = AsyncMock(return_value=async_iter([]))

    # Fused results with special-character IDs
    fused = [
        {"id": "A,1", "content": "alpha", "_fused_score": 0.9},
        {"id": "B'2", "content": "bravo", "_fused_score": 0.8},
        {"id": "C", "content": "charlie", "_fused_score": 0.7},
    ]

    # Execute
    await provider._semantic_rerank("test query", fused)

    # Assert call and inspect filter
    provider._rerank_client.search.assert_awaited_once()
    call_kwargs = provider._rerank_client.search.call_args.kwargs
    filter_query = call_kwargs.get("filter")

    assert filter_query is not None, "Expected 'filter' kwarg on rerank search() call"

    # The filter should use OR clauses with properly escaped single quotes
    expected_clauses = [
        "id eq 'A,1'",
        "id eq 'B''2'",  # single quote must be doubled in OData
        "id eq 'C'",
    ]

    for clause in expected_clauses:
        assert clause in filter_query, f"Missing clause: {clause}\nGot: {filter_query}"

    # Ensure it's not using search.in() which doesn't handle special chars properly
    assert "search.in(" not in filter_query, (
        f"Filter should use 'OR' clauses, not search.in().\nGot: {filter_query}"
    )

    # The filter should be a proper OR construction
    assert " or " in filter_query.lower(), (
        f"Filter should contain OR operators.\nGot: {filter_query}"
    )
