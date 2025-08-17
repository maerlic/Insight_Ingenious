"""Test suite for the knowledge base agent conversation flow.

This module contains tests for the `ConversationFlow` implementation of the
knowledge base agent. It verifies the agent's core logic, including its
ability to use different backend data sources (Azure Search, local ChromaDB),
handle streaming responses correctly, and manage errors gracefully.

The tests rely heavily on mocking and patching to isolate the conversation
flow from external dependencies like language models, Azure services, and
local file systems.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncGenerator, NoReturn
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.models.chat import ChatRequest
from ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent import (
    ConversationFlow,
)

# --- Simple FunctionTool shim and AssistantAgent mocks ---


@dataclass
class FunctionToolShim:
    """A simple shim to stand in for the real FunctionTool."""

    func: Any
    description: str = ""


class MockAssistantResponse:
    """Mocks the response object from an AssistantAgent."""

    def __init__(self, content: str) -> None:
        """Initializes the mock response with specific content."""
        self.chat_message = SimpleNamespace(content=content)


class MockAssistantAgent:
    """Mocks the AssistantAgent to simulate tool calls and streaming."""

    def __init__(
        self,
        name: str,
        system_message: str,
        model_client: Any,
        tools: list[Any] | None = None,
        **_: Any,
    ) -> None:
        """
        Initializes the mock agent.

        Stores configuration and tools to simulate agent behavior without
        making real API calls.
        """
        self.name = name
        self.system_message = system_message
        self.model_client = model_client
        self.tools = tools or []

    async def on_messages(
        self, messages: list[Any], cancellation_token: Any | None = None
    ) -> MockAssistantResponse:
        """
        Simulates the agent receiving messages and executing a tool.

        It parses the user question and, if a tool is present, calls its
        function with the question to generate a response.
        """
        # Extract the actual question text from the single user message string
        user_text: str = messages[0].content
        q = (
            user_text.split("User question:", 1)[-1].strip()
            if "User question:" in user_text
            else user_text.strip()
        )

        # Use first tool if present to simulate a search
        out: Any = "no-tool"
        if self.tools:
            tool = self.tools[0]
            fn = getattr(tool, "func", tool)
            result = fn(q)
            out = await result if inspect.isawaitable(result) else result
        return MockAssistantResponse(out)

    def run_stream(
        self, task: Any, cancellation_token: Any | None = None
    ) -> AsyncGenerator[SimpleNamespace, None]:
        """
        Simulates the streaming response generation process.

        Yields a fake content chunk and a fake token usage update to mimic
        the behavior of a real streaming agent.
        """

        async def _gen() -> AsyncGenerator[SimpleNamespace, None]:
            # Yield a content "chunk"
            yield SimpleNamespace(content="chunk-1", usage=None)
            # Then a token usage update
            yield SimpleNamespace(
                content=None,
                usage=SimpleNamespace(total_tokens=42, completion_tokens=10),
            )

        return _gen()


@pytest.fixture(autouse=True)
def patch_tool_and_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patches FunctionTool, AssistantAgent, and Azure SDK modules for all tests.

    This fixture ensures that tests run in an isolated environment by replacing
    key external dependencies with lightweight shims and mocks. It also mocks
    the necessary parts of the Azure SDK to prevent network calls during
    preflight checks.
    """
    # Patch FunctionTool to our shim
    monkeypatch.setattr(
        "ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent.FunctionTool",
        FunctionToolShim,
    )
    # Patch AssistantAgent in the module where used
    monkeypatch.setattr(
        "ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent.AssistantAgent",
        MockAssistantAgent,
    )

    # Minimal Azure SDK surface used by KB preflight
    import types

    core_creds = types.ModuleType("azure.core.credentials")

    class _Cred:
        """Minimal mock for AzureKeyCredential."""

        def __init__(self, key: str) -> None:
            """Stores the key."""
            self.key = key

    core_creds.AzureKeyCredential = _Cred

    aio = types.ModuleType("azure.search.documents.aio")

    class _Client:
        """Minimal mock for SearchClient."""

        def __init__(self, *, endpoint: str, index_name: str, credential: Any) -> None:
            """Initializes the mock client, performing no actions."""
            pass

        async def get_document_count(self) -> int:
            """Simulates finding one document in the index."""
            return 1

        async def close(self) -> None:
            """Simulates closing the client connection."""
            pass

    aio.SearchClient = _Client

    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(sys.modules, "azure.core", types.ModuleType("azure.core"))
    monkeypatch.setitem(sys.modules, "azure.core.credentials", core_creds)
    monkeypatch.setitem(sys.modules, "azure.search", types.ModuleType("azure.search"))
    monkeypatch.setitem(
        sys.modules,
        "azure.search.documents",
        types.ModuleType("azure.search.documents"),
    )
    monkeypatch.setitem(sys.modules, "azure.search.documents.aio", aio)


@pytest.fixture
def mock_model_client(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Mocks the AOAI client creation to inject a fake client.

    This allows tests to verify interactions with the model client, such as
    checking if its `close` method was called, without creating a real client.
    """
    # Provide an object with an async close() we can assert against
    fake_client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        "ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent.create_aoai_chat_completion_client_from_config",
        lambda _: fake_client,
    )
    return fake_client


def make_config(azure: bool = True) -> SimpleNamespace:
    """
    Creates a minimal configuration object for tests.

    This helper generates a SimpleNamespace object that mimics the application's
    configuration, including model settings and optional Azure Search services.
    """
    # Minimal config with models[0] and optional azure_search_services
    model0 = SimpleNamespace(model="gpt-4o")
    cfg = SimpleNamespace(models=[model0])
    if azure:
        svc = SimpleNamespace(endpoint="https://s.net", key="key", index_name="idx")
        cfg.azure_search_services = [svc]
    else:
        cfg.azure_search_services = []
    return cfg


@pytest.mark.asyncio
async def test_kb_agent_uses_azure_backend_and_closes_client(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, mock_model_client: Any
) -> None:
    """
    Tests the happy path where the agent uses Azure Search and closes resources.

    This test verifies that when Azure Search is configured, the agent
    correctly uses the AzureSearchProvider to retrieve data and ensures
    that the underlying model client is properly closed after the request.
    """

    # Patch provider to return cleaned chunks
    class FakeProvider:
        """A mock for AzureSearchProvider for this specific test."""

        def __init__(self, *_: Any) -> None:
            """Initializes the fake provider."""
            pass

        async def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
            """Simulates retrieving a document from Azure Search."""
            return [{"id": "A", "content": "Alpha", "_final_score": 3.2, "title": "T"}]

        async def close(self) -> None:
            """Simulates closing the provider."""
            pass

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.AzureSearchProvider",
        FakeProvider,
    )

    # Stub chat service memory
    chat_history_repo = SimpleNamespace(
        get_thread_messages=AsyncMock(
            return_value=[SimpleNamespace(role="user", content="Hello world")]
        )
    )
    chat_service = SimpleNamespace(chat_history_repository=chat_history_repo)

    flow = ConversationFlow.__new__(ConversationFlow)
    flow._config = make_config(azure=True)
    flow._chat_service = chat_service
    flow._memory_path = str(tmp_path)
    # Provide explicit paths because __init__ was bypassed
    flow._kb_path = str(
        tmp_path / "knowledge_base"
    )  # does not exist yet → "empty" message
    flow._chroma_path = str(tmp_path / "chroma_db")

    req = ChatRequest(user_prompt="what is alpha?", thread_id="t1")
    resp = await flow.get_conversation_response(req)

    assert "Alpha" in resp.agent_response  # came from AzureSearchProvider.retrieve()
    # model client closed
    mock_model_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_kb_agent_chroma_fallback_empty_dir_message(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, mock_model_client: Any
) -> None:
    """
    Tests fallback to ChromaDB when Azure is not configured.

    This test verifies that if Azure Search is unavailable, the agent falls
    back to the local ChromaDB vector store. It specifically checks that the
    correct message is returned when the knowledge base directory is empty.
    """
    # No Azure -> Chroma fallback path; ensure policy allows local retrieval
    monkeypatch.setenv("KB_POLICY", "local_only")

    flow = ConversationFlow.__new__(ConversationFlow)
    flow._config = make_config(azure=False)
    flow._chat_service = None
    flow._memory_path = str(tmp_path)
    # Provide default paths in case of any local fallback path execution
    flow._kb_path = os.path.join(str(tmp_path), "knowledge_base")
    flow._chroma_path = os.path.join(str(tmp_path), "chroma_db")

    # If chromadb ever gets imported in this path, ensure it's available as a fake module
    class FakeChroma:
        """A mock for the chromadb library."""

        class PersistentClient:
            """A mock for the chromadb persistent client."""

            def __init__(self, path: str) -> None:
                """Initializes the mock client."""
                pass

            def get_collection(self, name: str) -> NoReturn:
                """Simulates a collection not being found."""
                raise Exception("no coll")

            def create_collection(self, name: str) -> SimpleNamespace:
                """Simulates creating a new, empty collection."""
                return SimpleNamespace(add=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "chromadb", FakeChroma)

    req = ChatRequest(user_prompt="anything", thread_id=None)
    resp = await flow.get_conversation_response(req)
    # The assistant returns the tool output string
    assert "Knowledge base directory is empty" in resp.agent_response
    mock_model_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_streaming_sequence_and_error_handling(
    monkeypatch: pytest.MonkeyPatch, mock_model_client: Any
) -> None:
    """
    Tests the sequence of streamed chunks and verifies error handling.

    This test checks two scenarios:
    1. The normal streaming flow produces the expected sequence of chunk types
       (status, content, token_count, final).
    2. An error raised during streaming is handled gracefully, resulting in an
       error chunk being sent to the client.
    """
    flow = ConversationFlow.__new__(ConversationFlow)
    flow._config = make_config(azure=False)
    flow._chat_service = None
    flow._memory_path = "."

    # Normal streaming
    chunks = []
    async for ch in flow.get_streaming_conversation_response(
        ChatRequest(user_prompt="q", thread_id=None)
    ):
        chunks.append(ch)

    # Expect: status "Searching...", status "Generating...", a content chunk, a token_count chunk, and a final
    kinds = [c.chunk_type for c in chunks]
    assert kinds[:2] == ["status", "status"]
    assert "content" in kinds
    assert "token_count" in kinds
    assert kinds[-1] == "final"
    # Final includes token counts
    assert chunks[-1].token_count == 42
    assert chunks[-1].max_token_count == 10

    # Error path: patch AssistantAgent.run_stream to raise
    class BadAgent(MockAssistantAgent):
        """A mock agent that raises an error during streaming."""

        def run_stream(
            self, task: Any, cancellation_token: Any | None = None
        ) -> AsyncGenerator[None, None]:
            """Simulates a failure during the streaming process."""

            async def _bad() -> AsyncGenerator[None, None]:
                raise RuntimeError("boom")
                yield  # pragma: no cover

            return _bad()

    monkeypatch.setattr(
        "ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent.AssistantAgent",
        BadAgent,
    )

    chunks2 = []
    async for ch in flow.get_streaming_conversation_response(
        ChatRequest(user_prompt="q", thread_id=None)
    ):
        chunks2.append(ch)
    assert (
        chunks2[-1].chunk_type == "final" or chunks2[-1].chunk_type == "error"
    )  # final still yielded after error message
    # Ensure an error content chunk is present
    assert any(("Error during streaming" in (c.content or "")) for c in chunks2)


# ------- AzureSearchProvider focused tests (cleaning + rerank fallback + answer delegation) -------


@pytest.mark.asyncio
async def test_azure_provider_retrieve_cleans_and_reranks(
    monkeypatch: pytest.MonkeyPatch, async_iter: Any
) -> None:
    """
    Tests AzureSearchProvider's retrieve method, focusing on data cleaning.

    This test verifies that after retrieving and reranking documents, the provider
    cleans the results by removing internal fields (like vectors and intermediate
    scores) before returning them.
    """
    from ingenious.config.main_settings import IngeniousSettings
    from ingenious.config.models import AzureSearchSettings, ModelSettings
    from ingenious.services.azure_search.provider import AzureSearchProvider

    # Build settings quickly
    settings = IngeniousSettings.model_construct()
    settings.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="embed",
            api_key="k",
            base_url="https://oai",
        ),
        ModelSettings(
            model="gpt-4o", deployment="chat", api_key="k", base_url="https://oai"
        ),
    ]
    settings.azure_search_services = [
        AzureSearchSettings(
            service="svc", endpoint="https://s.net", key="sk", index_name="idx"
        )
    ]

    # Mock pipeline pieces and reranker client
    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(
        return_value=[{"id": "A", "_retrieval_score": 1.0}]
    )
    mock_pipeline.retriever.search_vector = AsyncMock(
        return_value=[{"id": "A", "_retrieval_score": 0.9, "vector": [0.1, 0.2]}]
    )
    mock_pipeline.fuser.fuse = AsyncMock(
        return_value=[
            {"id": "A", "_fused_score": 0.8, "vector": [0.2, 0.3], "@search.score": 1.0}
        ]
    )

    # Reranker returns a new score
    fake_rerank = [{"id": "A", "@search.reranker_score": 3.5, "content": "Alpha"}]

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
    )
    fake_client = MagicMock()
    fake_client.search = AsyncMock(return_value=async_iter(fake_rerank))
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_client,
    )
    # Ensure QueryType present
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
    )

    prov = AzureSearchProvider(settings)
    out = await prov.retrieve("q", top_k=1)
    assert len(out) == 1
    # Cleaned: vector/internal fields removed; but id/content retained
    assert "vector" not in out[0]
    assert "@search.score" not in out[0]
    assert "_fused_score" not in out[0]
    assert out[0]["id"] == "A"
    await prov.close()


@pytest.mark.asyncio
async def test_azure_provider_rerank_fallback_when_ids_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Tests the provider's fallback when retrieved documents lack IDs for reranking.

    If the initial retrieval step returns documents without an 'id' field,
    the semantic reranking step should be skipped. This test confirms that
    the provider gracefully falls back to using the fused results.
    """
    from ingenious.config.main_settings import IngeniousSettings
    from ingenious.config.models import AzureSearchSettings, ModelSettings
    from ingenious.services.azure_search.provider import AzureSearchProvider

    settings = IngeniousSettings.model_construct()
    settings.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="embed",
            api_key="k",
            base_url="https://oai",
        ),
        ModelSettings(
            model="gpt-4o", deployment="chat", api_key="k", base_url="https://oai"
        ),
    ]
    settings.azure_search_services = [
        AzureSearchSettings(
            service="svc", endpoint="https://s.net", key="sk", index_name="idx"
        )
    ]

    mock_pipeline = MagicMock()
    mock_pipeline.retriever.search_lexical = AsyncMock(
        return_value=[{"X": "no-id", "_retrieval_score": 1.0}]
    )
    mock_pipeline.retriever.search_vector = AsyncMock(return_value=[])
    mock_pipeline.fuser.fuse = AsyncMock(
        return_value=[{"X": "no-id", "_fused_score": 0.7}]
    )

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
    )
    fake_client = MagicMock()
    fake_client.search = AsyncMock()  # should not be called due to missing IDs
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
    )

    prov = AzureSearchProvider(settings)
    out = await prov.retrieve("q", top_k=1)
    # Should fallback to fused scores and then clean
    assert len(out) == 1
    assert out[0].get("_final_score") is None  # _final_score is removed by cleaning
    await prov.close()


@pytest.mark.asyncio
async def test_azure_provider_answer_delegates_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Tests that the provider's answer method delegates to the search pipeline.

    This ensures that the `answer` method, when enabled, correctly calls the
    underlying pipeline's `get_answer` method to generate a synthesized answer
    from search results.
    """
    from ingenious.config.main_settings import IngeniousSettings
    from ingenious.config.models import AzureSearchSettings, ModelSettings
    from ingenious.services.azure_search.provider import AzureSearchProvider

    settings = IngeniousSettings.model_construct()
    settings.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="embed",
            api_key="k",
            base_url="https://oai",
        ),
        ModelSettings(
            model="gpt-4o", deployment="chat", api_key="k", base_url="https://oai"
        ),
    ]
    settings.azure_search_services = [
        AzureSearchSettings(
            service="svc", endpoint="https://s.net", key="sk", index_name="idx"
        )
    ]

    mock_pipeline = MagicMock()
    mock_pipeline.get_answer = AsyncMock(
        return_value={"answer": "A", "source_chunks": []}
    )

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda cfg: mock_pipeline,
    )
    fake_client = MagicMock()
    fake_client.search = AsyncMock()
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.QueryType",
        SimpleNamespace(SEMANTIC="semantic"),
    )

    # answer() requires opt-in now
    prov = AzureSearchProvider(settings, enable_answer_generation=True)
    ans = await prov.answer("q")
    assert ans["answer"] == "A"
    await prov.close()
