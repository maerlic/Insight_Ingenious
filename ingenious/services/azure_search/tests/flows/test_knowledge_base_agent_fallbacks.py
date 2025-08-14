# tests/flows/test_knowledge_base_agent_fallbacks.py
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent import (
    ConversationFlow,
)


@pytest.mark.asyncio
async def test_kb_agent_azure_runtime_failure_falls_back_to_chroma(
    mock_ingenious_settings,
):
    """
    P0: Verify _search_knowledge_base falls back to ChromaDB when AzureSearchProvider raises a runtime exception.
    """

    # Minimal parent service required by IConversationFlow.__init__
    class _ParentSvc:
        def __init__(self, cfg):
            self.config = cfg
            self.conversation_flow = "knowledge_base_agent"

        chat_history_repository = MagicMock()
        openai_service = None  # optional; some flows may read this

    # Build a minimal config that has chat_history.memory_path
    cfg = NS(
        chat_history=NS(memory_path="/tmp"),
        # Optional fields that might be read later:
        models=NS(),
        web=NS(streaming_chunk_size=100),
    )

    # Avoid real memory manager initialization in IConversationFlow.__init__
    with patch(
        "ingenious.services.memory_manager.get_memory_manager", return_value=MagicMock()
    ):
        flow = ConversationFlow(parent_multi_agent_chat_service=_ParentSvc(cfg))

    # Ensure a simple local path used by the Chroma fallback
    flow._memory_path = "/tmp"

    # Mock the AzureSearchProvider to fail
    mock_provider_instance = AsyncMock()
    mock_provider_instance.retrieve.side_effect = RuntimeError(
        "Azure Connection Failed"
    )
    mock_provider_instance.close = AsyncMock()

    # Mock ChromaDB to succeed (the fallback)
    mock_chroma_client = MagicMock()
    mock_chroma_collection = MagicMock()
    mock_chroma_collection.query.return_value = {
        "documents": [["Fallback result from ChromaDB"]]
    }
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    with (
        patch(
            "ingenious.services.azure_search.provider.AzureSearchProvider",
            return_value=mock_provider_instance,
        ),
        patch(
            "chromadb.PersistentClient",
            return_value=mock_chroma_client,
        ),
    ):
        result = await flow._search_knowledge_base(
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
