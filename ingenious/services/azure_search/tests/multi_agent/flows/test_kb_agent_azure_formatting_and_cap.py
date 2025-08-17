"""Tests Azure Search formatting and configuration for the knowledge base agent.

This module contains unit tests for the Azure Search integration within the
knowledge base agent's conversation flow. It specifically verifies behaviors
related to the formatting of search results, such as content truncation and
the use of separators, as well as configuration fallbacks like using an
environment variable for the default search index.

The tests use monkeypatching to mock the Azure SDKs and the internal search
provider, allowing the agent's logic to be tested in isolation without any
real network dependencies.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import ingenious.services.chat_services.multi_agent.conversation_flows.knowledge_base_agent.knowledge_base_agent as kb


def install_azure_sdk_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks the Azure SDK modules to avoid runtime import errors.

    This function creates and injects fake `azure.core.credentials` and
    `azure.search.documents.aio` modules into `sys.modules`. This allows
    the knowledge base agent, which depends on these modules, to be
    instantiated and tested without requiring the actual Azure SDKs to be
    installed.
    """

    class _Cred:
        def __init__(self, key: str) -> None:
            self.key = key

    class _Client:
        def __init__(
            self, *, endpoint: str, index_name: str, credential: Any
        ) -> None: ...
        async def get_document_count(self) -> int:
            return 1

        async def close(self) -> None:
            pass

    cred = types.ModuleType("azure.core.credentials")
    cred.AzureKeyCredential = _Cred  # type: ignore[attr-defined]
    aio = types.ModuleType("azure.search.documents.aio")
    aio.SearchClient = _Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "azure.core.credentials", cred)
    monkeypatch.setitem(sys.modules, "azure.search.documents.aio", aio)


def provider_with(
    results: List[Dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mocks the internal AzureSearchProvider to return a fixed set of results.

    This allows tests to provide controlled, predictable search results to the
    knowledge base agent's search method, isolating the agent's formatting
    and processing logic from the actual data retrieval implementation.
    """
    prov_mod = types.ModuleType("ingenious.services.azure_search.provider")

    class AzureSearchProvider:
        def __init__(
            self, *_args: Any, **_kwargs: Any
        ) -> None:  # accept config or nothing
            pass

        async def retrieve(self, *a: Any, **k: Any) -> List[Dict[str, str]]:
            return results

        async def close(self) -> None:
            pass

    prov_mod.AzureSearchProvider = AzureSearchProvider  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "ingenious.services.azure_search.provider", prov_mod
    )


@pytest.mark.asyncio
async def test_azure_snippet_cap_truncates_snippet_and_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that KB_AZURE_SNIPPET_CAP truncates search result snippets and content.

    This test ensures that when the environment variable for capping snippet length
    is set, both the 'snippet' and 'content' fields of the search results are
    correctly truncated to the specified length. This is important for controlling
    the size of the context fed to the language model.
    """
    install_azure_sdk_ok(monkeypatch)
    # Long snippet and content
    doc: Dict[str, str] = {
        "id": "1",
        "title": "T1",
        "snippet": "ABCDEFGHIJKL",
        "content": "abcdefghijklmnop",
    }
    provider_with([doc], monkeypatch)

    os.environ["KB_AZURE_SNIPPET_CAP"] = "10"

    flow: kb.ConversationFlow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(
        models=[SimpleNamespace(model="gpt-4o")],
        azure_search_services=[
            SimpleNamespace(endpoint="https://s", key="k", index_name="idx")
        ],
    )
    flow._chat_service = None
    flow._memory_path = str(tmp_path)

    out: str = await flow._search_knowledge_base(
        "q", use_azure_search=True, top_k=3, logger=None
    )
    # capped to 10 chars each
    assert "ABCDEFGHIJ" in out
    assert "abcdefghij" in out


@pytest.mark.asyncio
async def test_multiple_results_include_separators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks that multiple search results are formatted with a separator.

    This ensures that when more than one document is returned from the knowledge
    base search, a clear separator is inserted between them. This helps in
    structuring the context for the language model, making it easier to
    distinguish between different sources of information.
    """
    install_azure_sdk_ok(monkeypatch)
    provider_with(
        [
            {"id": "1", "title": "A", "snippet": "s1", "content": "c1"},
            {"id": "2", "title": "B", "snippet": "s2", "content": "c2"},
        ],
        monkeypatch,
    )
    flow: kb.ConversationFlow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(
        models=[SimpleNamespace(model="gpt-4o")],
        azure_search_services=[
            SimpleNamespace(endpoint="https://s", key="k", index_name="idx")
        ],
    )
    flow._chat_service = None
    flow._memory_path = str(tmp_path)

    out: str = await flow._search_knowledge_base(
        "q", use_azure_search=True, top_k=2, logger=None
    )
    assert "\n\n---\n\n" in out  # separator present


@pytest.mark.asyncio
async def test_env_default_index_logs_info_when_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies an info message is logged when the default index env var is used.

    This test confirms that if the Azure Search configuration lacks an explicit
    index name, the system falls back to the `AZURE_SEARCH_DEFAULT_INDEX`
    environment variable and logs an informational message. This logging is
    crucial for transparency and debugging configuration issues.
    """
    install_azure_sdk_ok(monkeypatch)
    provider_with(
        [{"id": "1", "title": "T", "snippet": "S", "content": "C"}], monkeypatch
    )

    # index missing -> use env default and log INFO
    os.environ["AZURE_SEARCH_DEFAULT_INDEX"] = "docs"

    flow: kb.ConversationFlow = kb.ConversationFlow.__new__(kb.ConversationFlow)
    flow._config = SimpleNamespace(
        models=[SimpleNamespace(model="gpt-4o")],
        azure_search_services=[
            SimpleNamespace(endpoint="https://s", key="k", index_name="")
        ],
    )
    flow._chat_service = None
    flow._memory_path = str(tmp_path)

    caplog.set_level("INFO")
    logger: logging.Logger = logging.getLogger("kb-test")
    _: str = await flow._search_knowledge_base(
        "q", use_azure_search=True, top_k=1, logger=logger
    )
    assert any(
        "using env AZURE_SEARCH_DEFAULT_INDEX" in (r.getMessage() or "")
        for r in caplog.records
    )
