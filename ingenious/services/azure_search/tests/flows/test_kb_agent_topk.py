import logging
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

# Import the module under test (KB Agent)
import ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent as kb

# ──────────────────────────────────────────────────────────────────────────────
# Test doubles / small helpers
# ──────────────────────────────────────────────────────────────────────────────


class AcceptingLogHandler(logging.Handler):
    """A logging handler that accepts arbitrary kwargs in __init__ (mimics tracker)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        pass


class DummyLLMClient:
    async def close(self) -> None:
        pass


class DummyFunctionTool:
    """Mimics a tool wrapper; stores a callable and forwards calls."""

    def __init__(self, func, description: str = "") -> None:
        self.function = func
        self.description = description

    async def __call__(self, *args, **kwargs):
        return await self.function(*args, **kwargs)

    # Some frameworks call invoke(), keep it for safety
    async def invoke(self, *args, **kwargs):
        return await self.function(*args, **kwargs)


class DummyAssistantAgent:
    """Very small assistant that just calls the first tool with a search query."""

    def __init__(
        self,
        name: str,
        system_message: str,
        model_client,
        tools,
        reflect_on_tool_use: bool = True,
    ) -> None:
        self._name = name
        self._system_message = system_message
        self._client = model_client
        self._tools = tools

    async def on_messages(self, messages, cancellation_token=None):
        # Pull the question text from the message content.
        content = getattr(messages[0], "content", "")
        # The agent sends "Context: ...\n\nUser question: <q>"
        if "User question:" in content:
            q = content.split("User question:", 1)[1].strip()
        else:
            q = content.strip()

        # Use the first tool with the parsed query
        data = await self._tools[0].function(
            q
        )  # our DummyFunctionTool keeps the original callable

        class _Msg:
            def __init__(self, c):
                self.content = c

        class _Resp:
            def __init__(self, c):
                self.chat_message = _Msg(c)

        return _Resp(data)

    # Not used by these tests, but kept for completeness
    def run_stream(self, task: str, cancellation_token=None):
        async def _gen():
            class _Msg:
                def __init__(self, c):
                    self.content = c

            yield _Msg("stream content")

        return _gen()


def install_minimal_autogen(monkeypatch):
    """Provide tiny autogen_core, autogen_core.tools, autogen_agentchat.messages modules."""
    core_mod = types.ModuleType("autogen_core")
    core_mod.EVENT_LOGGER_NAME = "autogen"

    class _CT:
        pass

    core_mod.CancellationToken = _CT
    monkeypatch.setitem(sys.modules, "autogen_core", core_mod)

    tools_mod = types.ModuleType("autogen_core.tools")

    class _FT:
        pass

    tools_mod.FunctionTool = _FT
    monkeypatch.setitem(sys.modules, "autogen_core.tools", tools_mod)

    agents_mod = types.ModuleType("autogen_agentchat.agents")

    class _Assistant:
        pass

    agents_mod.AssistantAgent = _Assistant
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", agents_mod)

    msgs_mod = types.ModuleType("autogen_agentchat.messages")

    class TextMessage:
        def __init__(self, content: str, source: str):
            self.content = content
            self.source = source

    msgs_mod.TextMessage = TextMessage
    monkeypatch.setitem(sys.modules, "autogen_agentchat.messages", msgs_mod)


def install_dummy_token_counter(monkeypatch):
    tc = types.ModuleType("ingenious.utils.token_counter")

    def _num_tokens_from_messages(msgs, model):
        return 0

    tc.num_tokens_from_messages = _num_tokens_from_messages
    monkeypatch.setitem(sys.modules, "ingenious.utils.token_counter", tc)


def install_memory_manager(monkeypatch):
    mm = types.ModuleType("ingenious.services.memory_manager")

    class _MM:
        async def close(self):
            pass

        async def maintain_memory(self, new_content, max_words):
            return None

    def get_memory_manager(config, path):
        return _MM()

    async def run_async_memory_operation(coro):
        return await coro

    mm.get_memory_manager = get_memory_manager
    mm.run_async_memory_operation = run_async_memory_operation
    monkeypatch.setitem(sys.modules, "ingenious.services.memory_manager", mm)


def install_fake_provider(
    monkeypatch, results=None, raise_exc: Exception | None = None
):
    """Install a fake AzureSearchProvider that records calls and optionally raises."""
    prov_mod = types.ModuleType("ingenious.services.azure_search.provider")
    calls: List[Dict[str, Any]] = []

    class AzureSearchProvider:
        def __init__(self, settings, enable_answer_generation=None) -> None:
            pass

        async def retrieve(self, query: str, top_k: int = 10, **kwargs):
            calls.append({"query": query, "top_k": top_k, "kwargs": kwargs})
            if raise_exc:
                raise raise_exc
            return results or []

        async def answer(self, query: str):
            return {"answer": "stub", "source_chunks": []}

        async def close(self) -> None:
            pass

    prov_mod.AzureSearchProvider = AzureSearchProvider
    monkeypatch.setitem(
        sys.modules, "ingenious.services.azure_search.provider", prov_mod
    )
    return calls


def install_fake_chromadb(monkeypatch, documents: Optional[List[str]] = None):
    """Stub chromadb to return canned documents (so fallback path is deterministic)."""
    chroma_mod = types.ModuleType("chromadb")

    class _Collection:
        def add(self, documents, ids):
            pass

        def query(self, query_texts, n_results: int):
            docs = documents or ["Chroma Fallback Doc"]
            return {"documents": [docs[:n_results]]}

    class _Client:
        def __init__(self, path: str):
            self._path = path

        def get_collection(self, name: str):
            return _Collection()

        def create_collection(self, name: str):
            return _Collection()

    chroma_mod.PersistentClient = _Client
    monkeypatch.setitem(sys.modules, "chromadb", chroma_mod)


def make_config(
    memory_path: str,
    *,
    endpoint="https://example.search.windows.net",
    key="real-key",
    index_name="idx",
):
    """Create a very small config object with just the fields the KB flow reads."""
    chat_history = SimpleNamespace(memory_path=memory_path)
    # Only 'model' field is read by token counter
    models = [SimpleNamespace(model="gpt-4o")]
    azure_service = SimpleNamespace(
        endpoint=endpoint,
        key=key,
        index_name=index_name,
        use_semantic_ranking=False,
        semantic_ranking=False,
        semantic_configuration=None,
        top_k_retrieval=20,
        top_n_final=5,
        id_field="id",
        content_field="content",
        vector_field="vector",
    )
    # The flow also accesses some unrelated fields on config, but we can keep them absent.
    cfg = SimpleNamespace(
        chat_history=chat_history,
        models=models,
        azure_search_services=[azure_service],
        # The KB flow references _chat_service.chat_history_repository sometimes;
        # we will build the parent service below with this repository.
    )
    return cfg


class DummyChatHistoryRepo:
    async def get_thread_messages(self, thread_id: str):
        # Return empty list so memory context is blank and doesn't affect assertions
        return []


class DummyParentService:
    def __init__(self, config):
        self.config = config
        self.chat_history_repository = DummyChatHistoryRepo()


# ──────────────────────────────────────────────────────────────────────────────
# Autouse fixture: safe patching for every test in THIS FILE
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _common_patches(monkeypatch):
    """
    Runs automatically for every test in this module.
    Provides minimal stubs for external deps and patches KB agent symbols.
    """
    install_minimal_autogen(monkeypatch)
    install_dummy_token_counter(monkeypatch)
    install_memory_manager(monkeypatch)

    # Patch the KB module directly (no global leak)
    monkeypatch.setattr(
        kb,
        "create_aoai_chat_completion_client_from_config",
        lambda cfg: DummyLLMClient(),
    )
    monkeypatch.setattr(kb, "LLMUsageTracker", AcceptingLogHandler)
    monkeypatch.setattr(kb, "FunctionTool", DummyFunctionTool)
    monkeypatch.setattr(kb, "AssistantAgent", DummyAssistantAgent)

    # Clean up any env that could influence tests
    for var in (
        "KB_MODE",
        "KB_TOPK_DIRECT",
        "KB_TOPK_ASSIST",
        "AZURE_SEARCH_DEFAULT_INDEX",
    ):
        monkeypatch.delenv(var, raising=False)

    yield


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kb_agent_uses_azure_search_when_configured(tmp_path, monkeypatch):
    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "Alpha"}],
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    req = kb.ChatRequest(user_prompt="q")
    resp = await flow.get_conversation_response(req)

    assert "Found relevant information from Azure AI Search" in resp.agent_response
    # Direct mode default → top_k=3
    assert len(calls) == 1
    assert calls[0]["top_k"] == 3


@pytest.mark.asyncio
async def test_kb_agent_assist_mode_topk_5(tmp_path, monkeypatch):
    os.environ["KB_MODE"] = "assist"

    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "Alpha"}],
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    req = kb.ChatRequest(user_prompt="q")
    resp = await flow.get_conversation_response(req)

    assert "Found relevant information from Azure AI Search" in resp.agent_response
    assert len(calls) == 1
    # Assist mode default → top_k=5
    assert calls[0]["top_k"] == 5


@pytest.mark.asyncio
async def test_kb_agent_default_index_from_env_when_missing(
    tmp_path, monkeypatch, caplog
):
    # Service has no index configured
    cfg = make_config(str(tmp_path), index_name="")
    # Env supplies default index
    os.environ["AZURE_SEARCH_DEFAULT_INDEX"] = "docs"

    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "Doc", "snippet": "Info", "content": "Alpha"}],
    )

    flow = kb.ConversationFlow(DummyParentService(cfg))
    caplog.set_level(logging.INFO)

    resp = await flow.get_conversation_response(kb.ChatRequest(user_prompt="q"))

    assert "Found relevant information from Azure AI Search" in resp.agent_response
    assert len(calls) == 1  # provider was used (no fallback)
    # No WARNING about empty KB directory or missing index (INFO is acceptable)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        "Knowledge base directory is empty" in (r.getMessage() or "") for r in warnings
    )
    assert not any("using fallback default" in (r.getMessage() or "") for r in warnings)


@pytest.mark.asyncio
async def test_kb_agent_azure_failure_falls_back_to_chroma_with_message(
    tmp_path, monkeypatch
):
    # Azure provider is importable but fails at runtime
    calls = install_fake_provider(
        monkeypatch, raise_exc=RuntimeError("503 Service Unavailable")
    )
    install_fake_chromadb(monkeypatch, documents=["C1", "C2"])

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    resp = await flow.get_conversation_response(kb.ChatRequest(user_prompt="alpha?"))

    assert "Found relevant information from ChromaDB" in resp.agent_response
    # Provider was attempted once
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_kb_agent_maps_titles_and_snippets_in_output(tmp_path, monkeypatch):
    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "Alpha"}],
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    resp = await flow.get_conversation_response(kb.ChatRequest(user_prompt="q"))
    out = resp.agent_response

    assert "Found relevant information from Azure AI Search" in out
    assert "[1] T (score=" in out  # allow blank score
    assert "S" in out  # snippet included
    # content is also preserved (agent now includes both snippet and content)
    assert "Alpha" in out

    assert len(calls) == 1
    assert calls[0]["top_k"] == 3


@pytest.mark.asyncio
async def test_kb_agent_request_override_topk_wins(tmp_path, monkeypatch):
    """Per-request top_k should override defaults (uses dynamic attribute)."""
    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "X"}],
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    # We can pass a simple object with attributes expected by the agent.
    class Req:
        user_prompt = "q"
        kb_top_k = 7  # direct override
        thread_id = None

    resp = await flow.get_conversation_response(Req())
    assert "Found relevant information from Azure AI Search" in resp.agent_response
    assert len(calls) == 1
    assert calls[0]["top_k"] == 7


@pytest.mark.asyncio
async def test_kb_agent_env_override_direct(tmp_path, monkeypatch):
    os.environ["KB_TOPK_DIRECT"] = "9"
    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "X"}],
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    resp = await flow.get_conversation_response(kb.ChatRequest(user_prompt="q"))
    assert "Found relevant information from Azure AI Search" in resp.agent_response
    assert len(calls) == 1
    assert calls[0]["top_k"] == 9


@pytest.mark.asyncio
async def test_kb_agent_env_override_assist(tmp_path, monkeypatch):
    os.environ["KB_MODE"] = "assist"
    os.environ["KB_TOPK_ASSIST"] = "11"

    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "X"}],
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    resp = await flow.get_conversation_response(kb.ChatRequest(user_prompt="q"))
    assert "Found relevant information from Azure AI Search" in resp.agent_response
    assert len(calls) == 1
    assert calls[0]["top_k"] == 11


@pytest.mark.asyncio
async def test_kb_agent_snippet_fallbacks_to_content_when_missing(
    tmp_path, monkeypatch
):
    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "content": "Only content"}],  # no snippet
    )

    cfg = make_config(str(tmp_path))
    flow = kb.ConversationFlow(DummyParentService(cfg))

    resp = await flow.get_conversation_response(kb.ChatRequest(user_prompt="q"))
    out = resp.agent_response

    assert "Found relevant information from Azure AI Search" in out
    assert "Only content" in out  # content is printed when snippet missing
    assert len(calls) == 1
    assert calls[0]["top_k"] == 3


@pytest.mark.asyncio
async def test_kb_agent_warns_when_no_env_default_index(tmp_path, monkeypatch, caplog):
    # index is missing and no env override → we should warn about fallback 'test-index'
    cfg = make_config(str(tmp_path), index_name="")
    calls = install_fake_provider(
        monkeypatch,
        results=[{"id": "1", "title": "T", "snippet": "S", "content": "X"}],
    )

    flow = kb.ConversationFlow(DummyParentService(cfg))
    caplog.set_level(logging.WARNING)

    _ = await flow.get_conversation_response(kb.ChatRequest(user_prompt="q"))

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("fallback default 'test-index'" in (m or "") for m in warnings)
    assert len(calls) == 1
