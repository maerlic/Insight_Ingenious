"""Test knowledge base agent configuration resolution logic.

This module contains tests to verify that the ConversationFlow for the
knowledge base agent correctly resolves settings like the number of search
results (`top_k`) and the retrieval mode. It specifically tests the
priority order: request parameters override environment variables, which in
turn override hardcoded defaults. It also validates fallback behavior for
invalid or missing configuration values.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List

import pytest

import ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent as kb

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@pytest.mark.asyncio
async def test_request_parameters_string_topk_overrides_env_and_defaults(
    tmp_path: Path,
) -> None:
    """Verify request `kb_top_k` parameter overrides environment and defaults.

    This test ensures that when `kb_top_k` is provided in the request parameters,
    its value is used, even if it's a string, superseding any value from
    environment variables or the hardcoded default.
    """
    flow: kb.ConversationFlow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(models=[SimpleNamespace(model="gpt-4o")])
    flow._chat_service = None
    flow._memory_path = str(tmp_path)
    # Provide default paths in case of any local fallback path execution
    flow._kb_path = os.path.join(str(tmp_path), "knowledge_base")
    flow._chroma_path = os.path.join(str(tmp_path), "chroma_db")

    class Req:
        """A mock request with a string 'kb_top_k' parameter."""

        user_prompt: str = "q"
        parameters: Dict[str, str] = {"kb_top_k": "12"}

    assert flow._get_top_k("direct", Req()) == 12
    assert flow._get_top_k("assist", Req()) == 12


def test_non_numeric_and_non_positive_topk_values_are_ignored(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Check that invalid 'kb_top_k' values are ignored and fallbacks are used.

    This test verifies that if the `kb_top_k` parameter in a request is
    non-numeric or non-positive (e.g., "abc" or "0"), it is disregarded.
    The system should then fall back to using the value from the environment
    variable (`KB_TOPK_DIRECT`) or, if that is not set, the default value.
    """
    flow: kb.ConversationFlow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(models=[SimpleNamespace(model="gpt-4o")])
    flow._chat_service = None
    flow._memory_path = str(tmp_path)
    # Provide default paths in case of any local fallback path execution
    flow._kb_path = os.path.join(str(tmp_path), "knowledge_base")
    flow._chroma_path = os.path.join(str(tmp_path), "chroma_db")

    class BadReq:
        """A mock request with a non-numeric 'kb_top_k' parameter."""

        user_prompt: str = "q"
        parameters: Dict[str, str] = {"kb_top_k": "abc"}  # ignored

    os.environ["KB_TOPK_DIRECT"] = "7"
    assert flow._get_top_k("direct", BadReq()) == 7

    # non-positive should be ignored -> defaults apply when no env set
    monkeypatch.delenv("KB_TOPK_DIRECT", raising=False)

    class ZeroReq:
        """A mock request with a non-positive 'top_k' parameter."""

        user_prompt: str = "q"
        parameters: Dict[str, str] = {"top_k": "0"}

    assert flow._get_top_k("direct", ZeroReq()) == 3  # default


@pytest.mark.asyncio
async def test_invalid_kb_mode_coerces_to_direct(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Ensure an invalid KB_MODE environment variable falls back to 'direct' mode.

    If the `KB_MODE` environment variable is set to an unsupported value,
    the conversation flow should default to 'direct' mode and use its
    corresponding `top_k` default (3), ensuring graceful degradation.
    """
    # If KB_MODE invalid, direct-mode defaults are used (top_k=3).
    os.environ["KB_MODE"] = "wEiRd"

    # Provider stub captures top_k
    prov_mod = types.ModuleType("ingenious.services.azure_search.provider")
    seen: Dict[str, Any] = {}

    class AzureSearchProvider:
        """A stub for the AzureSearchProvider to capture the `top_k` value."""

        def __init__(self, *a: Any, **k: Any) -> None:
            """Initialize the mock provider, accepting any arguments."""
            pass  # accept config

        async def retrieve(self, query: str, top_k: int = 99) -> List[Dict[str, str]]:
            """Record the `top_k` value used in the call and return a mock result."""
            seen["top_k"] = top_k
            return [{"id": "1", "title": "T", "snippet": "S", "content": "C"}]

        async def close(self) -> None:
            """Perform no-op cleanup."""
            pass

    prov_mod.AzureSearchProvider = AzureSearchProvider
    import types as _t

    class _Cred:
        """A mock for AzureKeyCredential."""

        def __init__(self, k: str) -> None:
            """Initialize with a key."""
            self.k = k

    class _Client:
        """A mock for azure.search.documents.aio.SearchClient."""

        def __init__(self, *, endpoint: str, index_name: str, credential: Any) -> None:
            """Initialize the mock client."""
            ...

        async def get_document_count(self) -> int:
            """Return a mock document count."""
            return 1

        async def close(self) -> None:
            """Perform no-op cleanup."""
            pass

    monkeypatch.setitem(
        sys.modules, "azure.core.credentials", _t.ModuleType("azure.core.credentials")
    )
    sys.modules["azure.core.credentials"].AzureKeyCredential = _Cred
    monkeypatch.setitem(
        sys.modules,
        "azure.search.documents.aio",
        _t.ModuleType("azure.search.documents.aio"),
    )
    sys.modules["azure.search.documents.aio"].SearchClient = _Client
    monkeypatch.setitem(
        sys.modules, "ingenious.services.azure_search.provider", prov_mod
    )

    # Minimal autogen + LLM tracker + client
    core = _t.ModuleType("autogen_core")
    core.EVENT_LOGGER_NAME = "autogen"

    class CancellationToken:
        """A mock CancellationToken."""

        ...

    core.CancellationToken = CancellationToken
    monkeypatch.setitem(sys.modules, "autogen_core", core)
    tools = _t.ModuleType("autogen_core.tools")
    tools.FunctionTool = object
    monkeypatch.setitem(sys.modules, "autogen_core.tools", tools)
    ag = _t.ModuleType("autogen_agentchat.agents")
    ag.AssistantAgent = object
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", ag)

    class DummyLLMClient:
        """A mock LLM client with a no-op close method."""

        async def close(self) -> None:
            """Perform no-op cleanup."""
            pass

    monkeypatch.setattr(
        kb, "create_aoai_chat_completion_client_from_config", lambda _: DummyLLMClient()
    )

    class Acc:
        """A mock accumulator."""

        def emit(self, r: Any) -> None:
            """Perform a no-op emit."""
            pass

    import logging

    class _H(logging.Handler):
        """A mock logging handler."""

        def emit(self, r: Any) -> None:
            """Perform a no-op emit."""
            pass

    monkeypatch.setattr(kb, "LLMUsageTracker", _H, raising=False)

    flow: kb.ConversationFlow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(
        models=[SimpleNamespace(model="gpt-4o")],
        azure_search_services=[
            SimpleNamespace(endpoint="https://s", key="k", index_name="idx")
        ],
    )
    flow._chat_service = None
    flow._memory_path = str(tmp_path)
    # Provide default paths in case of any local fallback path execution
    flow._kb_path = os.path.join(str(tmp_path), "knowledge_base")
    flow._chroma_path = os.path.join(str(tmp_path), "chroma_db")

    from ingenious.models.chat import ChatRequest

    resp: Any = await flow.get_conversation_response(ChatRequest(user_prompt="q"))
    assert "Found relevant information from Azure AI Search" in resp.agent_response
    assert seen.get("top_k") == 3  # direct default applies
