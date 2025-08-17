"""Test the fallback mechanisms for the knowledge base agent.

This module contains tests to verify that the ConversationFlow for the
knowledge base agent correctly handles failures of its primary data source
(Azure AI Search) and falls back to a secondary source (local ChromaDB).
The primary entry point is the test case that simulates a runtime error
from the Azure search provider.
"""

from __future__ import annotations

import os
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent import (
    ConversationFlow,
)


@pytest.mark.asyncio
async def test_kb_agent_azure_runtime_failure_falls_back_to_chroma(
    mock_ingenious_settings: MagicMock,
) -> None:
    """
    P0: Verify _search_knowledge_base falls back to ChromaDB when AzureSearchProvider raises a runtime exception.
    """

    # Minimal parent service required by IConversationFlow.__init__
    class _ParentSvc:
        """A minimal mock of the parent service needed by ConversationFlow."""

        chat_history_repository: MagicMock = MagicMock()
        openai_service: None = None  # optional; some flows may read this

        def __init__(self, cfg: NS) -> None:
            """Initialize the mock parent service with a configuration."""
            self.config = cfg
            self.conversation_flow = "knowledge_base_agent"

    # Build a minimal config that has chat_history.memory_path
    cfg: NS = NS(
        chat_history=NS(memory_path="/tmp"),
        # Optional fields that might be read later:
        models=NS(),
        web=NS(streaming_chunk_size=100),
    )
    # Provide minimal Azure service so preflight runs and the provider is attempted
    cfg.azure_search_services = [
        NS(endpoint="https://s.net", key="sk", index_name="idx")
    ]

    # Avoid real memory manager initialization in IConversationFlow.__init__
    flow: ConversationFlow
    with patch(
        "ingenious.services.memory_manager.get_memory_manager", return_value=MagicMock()
    ):
        flow = ConversationFlow(parent_multi_agent_chat_service=_ParentSvc(cfg))

    # Ensure a simple local path used by the Chroma fallback
    flow._memory_path = "/tmp"

    # Mock the AzureSearchProvider to fail
    mock_provider_instance: AsyncMock = AsyncMock()
    mock_provider_instance.retrieve.side_effect = RuntimeError(
        "Azure Connection Failed"
    )
    mock_provider_instance.close = AsyncMock()

    # Mock ChromaDB to succeed (the fallback)
    mock_chroma_client: MagicMock = MagicMock()
    mock_chroma_collection: MagicMock = MagicMock()
    mock_chroma_collection.query.return_value = {
        "documents": [["Fallback result from ChromaDB"]]
    }
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    # Minimal Azure SDK surface
    class _Cred:
        """A minimal mock for AzureKeyCredential."""

        def __init__(self, key: str) -> None:
            """Initialize the mock credential."""
            self.key = key

    class _Client:
        """A minimal mock for SearchClient."""

        def __init__(
            self, *, endpoint: str, index_name: str, credential: _Cred
        ) -> None:
            """Initialize the mock search client."""
            pass

        async def get_document_count(self) -> int:
            """Return a mock document count."""
            return 1

        async def close(self) -> None:
            """Mock the async close method."""
            pass

    # Ensure local KB dir exists so local fallback path executes
    os.makedirs("/tmp/knowledge_base", exist_ok=True)

    with (
        patch("azure.core.credentials.AzureKeyCredential", _Cred, create=True),
        patch("azure.search.documents.aio.SearchClient", _Client, create=True),
        patch(
            "ingenious.services.azure_search.provider.AzureSearchProvider",
            return_value=mock_provider_instance,
        ),
        patch(
            "chromadb.PersistentClient",
            return_value=mock_chroma_client,
        ),
        patch.dict(os.environ, {"KB_POLICY": "prefer_azure"}, clear=False),
    ):
        result: str = await flow._search_knowledge_base(
            search_query="test query",
            use_azure_search=True,
            top_k=3,
            logger=MagicMock(),
        )

    # Azure provider was tried and failed
    mock_provider_instance.retrieve.assert_called_once_with("test query", top_k=3)
    mock_provider_instance.close.assert_called_once()

    # Fallback was used
    mock_chroma_client.get_collection.assert_called_once()
    mock_chroma_collection.query.assert_called_once()

    assert "Found relevant information from ChromaDB:" in result
    assert "Fallback result from ChromaDB" in result
