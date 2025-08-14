# tests/multi_agent/test_tool_factory_fallbacks.py

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.chat_services.multi_agent.tool_factory import ToolFunctions


@pytest.mark.asyncio
async def test_tool_factory_aisearch_provider_failure_falls_back_to_mock():
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
        result_json = await ToolFunctions.aisearch(
            search_query="error query", index_name="test-idx"
        )

    mock_provider_instance.retrieve.assert_called()
    mock_provider_instance.close.assert_called()

    data = json.loads(result_json)
    assert data["@odata.count"] == 1
    assert "Mock search result" in data["value"][0]["content"]
    assert "error query" in data["value"][0]["content"]
    assert "test-idx" in data["value"][0]["content"]
