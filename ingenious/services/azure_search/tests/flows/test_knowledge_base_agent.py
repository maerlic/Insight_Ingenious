# tests/flows/test_knowledge_base_agent.py
import inspect
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.models.chat import ChatRequest
from ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent import (
    ConversationFlow,
)

# --- Simple FunctionTool shim and AssistantAgent mocks ---


@dataclass
class FunctionToolShim:
    func: Any
    description: str = ""


class MockAssistantResponse:
    def __init__(self, content: str):
        self.chat_message = SimpleNamespace(content=content)


class MockAssistantAgent:
    def __init__(self, name, system_message, model_client, tools=None, **_):
        self.name = name
        self.system_message = system_message
        self.model_client = model_client
        self.tools = tools or []

    async def on_messages(self, messages, cancellation_token=None):
        # Extract the actual question text from the single user message string
        user_text = messages[0].content
        q = (
            user_text.split("User question:", 1)[-1].strip()
            if "User question:" in user_text
            else user_text.strip()
        )

        # Use first tool if present to simulate a search
        out = "no-tool"
        if self.tools:
            tool = self.tools[0]
            fn = getattr(tool, "func", tool)
            result = fn(q)
            out = await result if inspect.isawaitable(result) else result
        return MockAssistantResponse(out)

    def run_stream(self, task, cancellation_token=None):
        async def _gen():
            # Yield a content "chunk"
            yield SimpleNamespace(content="chunk-1", usage=None)
            # Then a token usage update
            yield SimpleNamespace(
                content=None,
                usage=SimpleNamespace(total_tokens=42, completion_tokens=10),
            )

        return _gen()


@pytest.fixture(autouse=True)
def patch_tool_and_agent(monkeypatch):
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


@pytest.fixture
def mock_model_client(monkeypatch):
    # Provide an object with an async close() we can assert against
    fake_client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        "ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent.create_aoai_chat_completion_client_from_config",
        lambda _: fake_client,
    )
    return fake_client


def make_config(azure=True):
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
    tmp_path, monkeypatch, mock_model_client
):
    # Patch provider to return cleaned chunks
    class FakeProvider:
        def __init__(self, *_):
            pass

        async def retrieve(self, query, top_k=3):
            return [{"id": "A", "content": "Alpha", "_final_score": 3.2, "title": "T"}]

        async def close(self):
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

    req = ChatRequest(user_prompt="what is alpha?", thread_id="t1")
    resp = await flow.get_conversation_response(req)

    assert "Alpha" in resp.agent_response  # came from AzureSearchProvider.retrieve()
    # model client closed
    mock_model_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_kb_agent_chroma_fallback_empty_dir_message(
    tmp_path, monkeypatch, mock_model_client
):
    # No Azure -> Chroma fallback path
    flow = ConversationFlow.__new__(ConversationFlow)
    flow._config = make_config(azure=False)
    flow._chat_service = None
    flow._memory_path = str(tmp_path)

    # If chromadb ever gets imported in this path, ensure it's available as a fake module
    class FakeChroma:
        class PersistentClient:
            def __init__(self, path):
                pass

            def get_collection(self, name):
                raise Exception("no coll")

            def create_collection(self, name):
                return SimpleNamespace(add=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "chromadb", FakeChroma)

    req = ChatRequest(user_prompt="anything", thread_id=None)
    resp = await flow.get_conversation_response(req)
    # The assistant returns the tool output string
    assert "Knowledge base directory is empty" in resp.agent_response
    mock_model_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_streaming_sequence_and_error_handling(monkeypatch, mock_model_client):
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
        def run_stream(self, task, cancellation_token=None):
            async def _bad():
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
async def test_azure_provider_retrieve_cleans_and_reranks(monkeypatch, async_iter):
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
async def test_azure_provider_rerank_fallback_when_ids_missing(monkeypatch):
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
async def test_azure_provider_answer_delegates_to_pipeline(monkeypatch):
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

    prov = AzureSearchProvider(settings)
    ans = await prov.answer("q")
    assert ans["answer"] == "A"
    await prov.close()
