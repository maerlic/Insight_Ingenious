"""Test the AzureSearchProvider's resource lifecycle management.

This module contains tests to verify that the AzureSearchProvider correctly
manages the lifecycle of its internal components, such as Azure SDK clients.
The primary focus is ensuring that resources are properly initialized and,
more importantly, gracefully terminated to prevent resource leaks (e.g.,
dangling network connections). It validates the provider's teardown logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.azure_search.provider import AzureSearchProvider


@pytest.mark.asyncio
async def test_provider_close_calls_all_underlying_clients(
    mock_ingenious_settings: MagicMock,
) -> None:
    """Verify that closing the provider also closes its internal clients.

    This test ensures that the `AzureSearchProvider.close` method correctly
    propagates the close call to all the underlying, managed clients (like
    the search pipeline and the reranking client). This is critical for
    graceful shutdown and preventing resource leaks.
    """
    # Mock the internal components that the provider manages
    mock_pipeline: AsyncMock = AsyncMock()
    mock_pipeline.close = AsyncMock()  # Ensure the close method itself is an AsyncMock

    mock_rerank_client: AsyncMock = AsyncMock()
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
