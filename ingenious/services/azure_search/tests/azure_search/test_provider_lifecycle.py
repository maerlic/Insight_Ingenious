# tests/azure_search/test_provider_lifecycle.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.azure_search.provider import AzureSearchProvider


@pytest.mark.asyncio
async def test_provider_close_calls_all_underlying_clients(mock_ingenious_settings):
    """
    P3: Verify AzureSearchProvider.close() awaits close() on the pipeline and the rerank client.
    """
    # Mock the internal components that the provider manages
    mock_pipeline = AsyncMock()
    mock_pipeline.close = AsyncMock()  # Ensure the close method itself is an AsyncMock

    mock_rerank_client = AsyncMock()
    mock_rerank_client.close = AsyncMock()

    # Initialize the provider, patching the internal components
    with (
        patch(
            "ingenious.services.azure_search.provider.build_search_pipeline",
            return_value=mock_pipeline,
        ),
        patch(
            "ingenious.services.azure_search.provider.make_search_client",
            return_value=mock_rerank_client,
        ),
        patch(
            "ingenious.services.azure_search.provider.build_search_config_from_settings",
            return_value=MagicMock(),
        ),
    ):
        provider = AzureSearchProvider(settings=mock_ingenious_settings)

        # Execute the close method
        await provider.close()

    # Assert that close was called on both managed components
    mock_pipeline.close.assert_called_once()
    mock_rerank_client.close.assert_called_once()
