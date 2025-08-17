"""Tests for AI Search tool configuration and parameter handling.

This module verifies the behavior of the `ToolFunctions.aisearch` tool,
specifically focusing on its handling of the `index_name` parameter. It ensures
that the tool correctly passes the index name to underlying search providers
that support it (`index` or `index_name` signature) and raises a
DeprecationWarning when a custom index is specified for a provider that does
not support it. This confirms backward compatibility and guides users toward
correct provider implementations.
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import (  # <-- ensure MagicMock is imported
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from ingenious.services.chat_services.multi_agent.tool_factory import ToolFunctions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_signature, index_name, expect_warning",
    [
        ("index_name", "custom-index", False),
        ("index", "custom-index", False),
        ("none", "default", False),
        ("none", "custom-index", True),
    ],
)
async def test_tool_factory_aisearch_index_name_parameter_handling_and_deprecation_warning(
    provider_signature: str, index_name: str, expect_warning: bool
) -> None:
    """Verify aisearch tool handles `index_name` and deprecation warnings correctly.

    This test checks that the `aisearch` tool function correctly interacts
    with different search provider signatures. It confirms that:
    1. The `index_name` parameter is passed to providers with `index_name` or `index`
       arguments in their `retrieve` method.
    2. A `DeprecationWarning` is issued if a non-default `index_name` is used
       with a provider whose `retrieve` method does not accept it.
    3. No warning is issued in valid scenarios.
    """
    assert not ToolFunctions._should_use_mock()

    mock_provider_instance: AsyncMock = AsyncMock()
    mock_provider_instance.close = AsyncMock()

    if provider_signature == "index_name":

        async def retrieve_with_index_name(
            query: str, top_k: int = 5, index_name: str = "default"
        ) -> list[dict[str, str]]:
            """Mock retrieve method that accepts `index_name`."""
            return [{"content": f"Result from {index_name}", "id": "1"}]

        mock_provider_instance.retrieve.side_effect = retrieve_with_index_name
    elif provider_signature == "index":

        async def retrieve_with_index(
            query: str, top_k: int = 5, index: str = "default"
        ) -> list[dict[str, str]]:
            """Mock retrieve method that accepts `index`."""
            return [{"content": f"Result from {index}", "id": "1"}]

        mock_provider_instance.retrieve.side_effect = retrieve_with_index
    else:

        async def retrieve_basic(query: str, top_k: int = 5) -> list[dict[str, str]]:
            """Mock retrieve method that does not accept an index parameter."""
            return [{"content": "Result from default", "id": "1"}]

        mock_provider_instance.retrieve.side_effect = retrieve_basic

    with (
        patch(
            "ingenious.services.azure_search.provider.AzureSearchProvider",
            return_value=mock_provider_instance,
        ),
        patch("ingenious.config.get_config", return_value=MagicMock()),
    ):
        result_json: str
        if expect_warning:
            # Expect a DeprecationWarning when index_name is unsupported and non-default
            with pytest.warns(
                DeprecationWarning, match="`index_name` is not supported"
            ):
                result_json = await ToolFunctions.aisearch(
                    search_query="test", index_name=index_name
                )
        else:
            # Assert NO DeprecationWarning emitted
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result_json = await ToolFunctions.aisearch(
                    search_query="test", index_name=index_name
                )
            assert not any(issubclass(m.category, DeprecationWarning) for m in w), (
                "Unexpected DeprecationWarning(s): "
                f"{[str(m.message) for m in w if issubclass(m.category, DeprecationWarning)]}"
            )

    assert json.loads(result_json)["@odata.count"] > 0
