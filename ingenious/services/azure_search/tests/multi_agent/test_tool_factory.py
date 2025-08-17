"""Tests for the ToolFunctions class, specifically the aisearch tool.

This module contains unit tests for the tool functions used within the multi-agent
chat service. It focuses on verifying the behavior of the `aisearch` tool,
ensuring it correctly handles both successful data retrieval and provider
failures by mocking its external dependencies.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ingenious.services.chat_services.multi_agent.tool_factory import ToolFunctions


@pytest.mark.asyncio
async def test_tool_factory_aisearch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify aisearch tool successfully formats results from the provider.

    This test mocks the AzureSearchProvider to return a predefined list of
    document chunks. It ensures the tool function correctly processes these
    chunks into the expected JSON format, including handling of titles and scores.
    """
    # Patch get_config to return stub
    monkeypatch.setattr(
        "ingenious.config.get_config",
        lambda: SimpleNamespace(),
    )

    # Patch AzureSearchProvider to return cleaned chunks
    class FakeProvider:
        """A mock search provider that returns a fixed set of results."""

        def __init__(self, *_: Any) -> None:
            """Initialize the mock provider."""
            pass

        async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            """Simulate retrieving documents, returning a fixed list."""
            return [
                {
                    "id": "1",
                    "content": "A",
                    "title": "T1",
                    "_final_score": 1.23,
                    "vector": [0.1],
                },  # vector ignored by tool
                {"id": "2", "content": "B", "_final_score": 0.5},
            ]

        async def close(self) -> None:
            """Simulate closing the provider connection."""
            pass

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.AzureSearchProvider",
        FakeProvider,
    )

    out: str = await ToolFunctions.aisearch("my query")
    data: dict[str, Any] = json.loads(out)
    assert data["@odata.count"] == 2
    assert data["value"][0]["@search.score"] == 1.23
    assert data["value"][0]["content"] == "A"
    assert data["value"][0]["title"] == "T1"  # uses title if present
    # second item title falls back to id
    assert data["value"][1]["title"] == "2"


@pytest.mark.asyncio
async def test_tool_factory_aisearch_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify aisearch returns a mock response when the provider fails.

    This test mocks the AzureSearchProvider to raise an exception during retrieval.
    It ensures that the aisearch tool catches the exception and returns a
    pre-canned, user-friendly mock search result instead of propagating the error.
    """
    monkeypatch.setattr(
        "ingenious.config.get_config",
        lambda: SimpleNamespace(),
    )

    class BadProvider:
        """A mock search provider that always fails."""

        def __init__(self, *_: Any) -> None:
            """Initialize the failing mock provider."""
            pass

        async def retrieve(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            """Simulate a failed retrieval by raising a RuntimeError."""
            raise RuntimeError("boom")

        async def close(self) -> None:
            """Simulate closing the provider connection."""
            pass

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.AzureSearchProvider",
        BadProvider,
    )

    out: str = await ToolFunctions.aisearch("q")
    data: dict[str, Any] = json.loads(out)
    # Now expects a mock response instead of an error
    assert "@odata.count" in data
    assert data["@odata.count"] == 1
    assert len(data["value"]) == 1
    assert "Mock search result" in data["value"][0]["content"]
    assert data["value"][0]["title"] == "Mock Document q"
