"""Create Azure AI service clients from application configuration.

This module provides factory functions to instantiate clients for Azure AI Search
and Azure OpenAI. It centralizes client creation and configuration, ensuring that
services are initialized consistently based on a shared `SearchConfig` object.
This simplifies dependency management and keeps service connection logic separate
from application business logic.

The main entry points are `make_search_client` and `make_async_openai_client`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from openai import AsyncAzureOpenAI

if TYPE_CHECKING:
    from .config import SearchConfig


def make_search_client(cfg: SearchConfig) -> SearchClient:
    """Create and configure an Azure AI Search client.

    This factory function encapsulates the logic for instantiating the asynchronous
    `SearchClient`. It uses the endpoint, index name, and credentials from the
    provided configuration object to ensure the client is correctly set up.
    """
    return SearchClient(
        endpoint=cfg.search_endpoint,
        index_name=cfg.search_index_name,
        credential=AzureKeyCredential(cfg.search_key.get_secret_value()),
        # Removed retry_policy - use Azure SDK defaults
    )


def make_async_openai_client(cfg: SearchConfig) -> AsyncAzureOpenAI:
    """Create and configure an asynchronous Azure OpenAI client.

    This factory function sets up the `AsyncAzureOpenAI` client for making API
    calls. It configures the client with the necessary Azure endpoint, API key,
    and API version, and sets a default retry policy.
    """
    # openai-py has its own retry; we set a small max_retries
    return AsyncAzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_key.get_secret_value(),
        api_version=cfg.openai_version,
        max_retries=3,
    )
