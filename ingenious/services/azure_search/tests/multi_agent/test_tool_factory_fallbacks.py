"""Test fallback mechanisms for the ToolFunctions factory.

This module contains tests to verify that the ToolFunctions class, which provides
tools for multi-agent systems, handles external service failures gracefully.
Specifically, it ensures that when a primary provider (like Azure Search) fails,
the tool falls back to a predefined mock or default behavior instead of
propagating the exception, thus maintaining system robustness.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.chat_services.multi_agent.tool_factory import ToolFunctions


@pytest.mark.asyncio
async def test_tool_factory_aisearch_provider_failure_falls_back_to_mock() -> None:
    """Verify aisearch falls back to a mock response on provider failure.

    This test simulates a scenario where the AzureSearchProvider fails to connect
    or retrieve data by raising a RuntimeError. It checks that the
    ToolFunctions.aisearch method catches this exception and returns a
    structured mock JSON response, ensuring the agent's workflow is not
    interrupted by transient provider issues.
    """
    mock_provider_instance = AsyncMock()
    mock_provider_instance.retrieve.side_effect = RuntimeError(
        "Provider Connection Failed"
    )
    mock_provider_instance.close = AsyncMock()

    with (
        patch(
            "ingenious.services.azure_search.provider.AzureSearchProvider",
            return_value=mock_provider_instance,
        ),
        patch("ingenious.config.get_config", return_value=MagicMock()),
    ):
        result_json: str = await ToolFunctions.aisearch(
            search_query="error query", index_name="test-idx"
        )

    mock_provider_instance.retrieve.assert_called()
    mock_provider_instance.close.assert_called()

    data: dict[str, Any] = json.loads(result_json)
    assert data["@odata.count"] == 1
    assert "Mock search result" in data["value"][0]["content"]
    assert "error query" in data["value"][0]["content"]
    assert "test-idx" in data["value"][0]["content"]
