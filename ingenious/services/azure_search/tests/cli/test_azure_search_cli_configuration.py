# tests/cli/test_azure_search_cli_configuration.py

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
    provider_signature, index_name, expect_warning
):
    assert not ToolFunctions._should_use_mock()

    mock_provider_instance = AsyncMock()
    mock_provider_instance.close = AsyncMock()

    if provider_signature == "index_name":

        async def retrieve_with_index_name(query, top_k=5, index_name="default"):
            return [{"content": f"Result from {index_name}", "id": "1"}]

        mock_provider_instance.retrieve = retrieve_with_index_name
    elif provider_signature == "index":

        async def retrieve_with_index(query, top_k=5, index="default"):
            return [{"content": f"Result from {index}", "id": "1"}]

        mock_provider_instance.retrieve = retrieve_with_index
    else:

        async def retrieve_basic(query, top_k=5):
            return [{"content": "Result from default", "id": "1"}]

        mock_provider_instance.retrieve = retrieve_basic

    with (
        patch(
            "ingenious.services.azure_search.provider.AzureSearchProvider",
            return_value=mock_provider_instance,
        ),
        patch("ingenious.config.get_config", return_value=MagicMock()),
    ):
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
                f"Unexpected DeprecationWarning(s): {[str(m.message) for m in w if issubclass(m.category, DeprecationWarning)]}"
            )

    assert json.loads(result_json)["@odata.count"] > 0
