# -*- coding: utf-8 -*-

"""
FILE TEST PLAN

    client_init:
        test_make_search_client_uses_secretstr
            - Stubs azure modules at import-time (sys.modules) so the client_init
              module builds without real Azure SDK.
            - Verifies SearchClient constructor args and that SecretStr secrets are unwrapped.

        test_make_async_openai_client_maps_version_and_max_retries
            - Stubs openai.AsyncAzureOpenAI at import-time and asserts that the
              function maps endpoint, key (unwrapped), api_version and max_retries=3.
"""

import importlib
import sys
import types

from pydantic import SecretStr

from ingenious.services.azure_search.config import SearchConfig


def _install_dummy_sdk_modules(monkeypatch):
    """
    Install minimal dummy modules/classes into sys.modules so that when
    ingenious.services.azure_search.client_init is (re)imported, it binds
    to these instead of real SDKs.
    Returns the dummy classes for assertion.
    """

    # --- azure.core.credentials.AzureKeyCredential ---------------------------
    class DummyAzureKeyCredential:
        def __init__(self, key: str):
            self.key = key

    # --- azure.search.documents.aio.SearchClient -----------------------------
    class DummySearchClient:
        def __init__(self, *, endpoint, index_name, credential, **kwargs):
            self.endpoint = endpoint
            self.index_name = index_name
            self.credential = credential
            # Accept any extra kwargs without requiring them
            self.extra_kwargs = kwargs

    # --- openai.AsyncAzureOpenAI --------------------------------------------
    class DummyAsyncAzureOpenAI:
        def __init__(self, *, azure_endpoint, api_key, api_version, max_retries):
            self.azure_endpoint = azure_endpoint
            self.api_key = api_key
            self.api_version = api_version
            self.max_retries = max_retries

    # Build module objects and register in sys.modules
    azure = types.ModuleType("azure")
    core = types.ModuleType("azure.core")
    credentials = types.ModuleType("azure.core.credentials")
    credentials.AzureKeyCredential = DummyAzureKeyCredential

    search = types.ModuleType("azure.search")
    documents = types.ModuleType("azure.search.documents")
    aio = types.ModuleType("azure.search.documents.aio")
    aio.SearchClient = DummySearchClient

    openai_mod = types.ModuleType("openai")
    openai_mod.AsyncAzureOpenAI = DummyAsyncAzureOpenAI

    # Install in sys.modules via monkeypatch (auto-restore per test)
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.core", core)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", credentials)
    monkeypatch.setitem(sys.modules, "azure.search", search)
    monkeypatch.setitem(sys.modules, "azure.search.documents", documents)
    monkeypatch.setitem(sys.modules, "azure.search.documents.aio", aio)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    return {
        "AzureKeyCredential": DummyAzureKeyCredential,
        "SearchClient": DummySearchClient,
        "AsyncAzureOpenAI": DummyAsyncAzureOpenAI,
    }


def _reload_client_init_with_dummies(monkeypatch):
    dummies = _install_dummy_sdk_modules(monkeypatch)
    # Now (re)load the module so it binds to our dummies
    import ingenious.services.azure_search.client_init as client_init

    client_init = importlib.reload(client_init)
    return client_init, dummies


def _make_cfg(**overrides) -> SearchConfig:
    data = dict(
        search_endpoint="https://s.example.net",
        search_key=SecretStr("search-secret"),
        search_index_name="my-index",
        openai_endpoint="https://oai.example.com",
        openai_key=SecretStr("openai-secret"),
        embedding_deployment_name="emb-deploy",
        generation_deployment_name="gen-deploy",
        # use default openai_version unless overridden
    )
    data.update(overrides)
    return SearchConfig(**data)


# RENAMED: was test_make_search_client_uses_retry_policy_and_secretstr
def test_make_search_client_uses_secretstr(monkeypatch):
    """
    Test that SearchClient is created with unwrapped SecretStr.
    (Removed retry_policy testing since we removed it from implementation)
    """
    client_init, d = _reload_client_init_with_dummies(monkeypatch)

    cfg = _make_cfg()
    sc = client_init.make_search_client(cfg)

    # Type and constructor args captured by our dummy
    assert isinstance(sc, d["SearchClient"])
    assert sc.endpoint == cfg.search_endpoint
    assert sc.index_name == cfg.search_index_name

    # SecretStr was unwrapped (not the SecretStr object itself)
    assert isinstance(sc.credential, d["AzureKeyCredential"])
    assert sc.credential.key == "search-secret"

    # No longer testing retry_policy since we removed it


def test_make_async_openai_client_maps_version_and_max_retries(monkeypatch):
    client_init, d = _reload_client_init_with_dummies(monkeypatch)

    cfg = _make_cfg(openai_version="2025-01-01")
    oc = client_init.make_async_openai_client(cfg)

    assert isinstance(oc, d["AsyncAzureOpenAI"])
    assert oc.azure_endpoint == cfg.openai_endpoint
    # SecretStr was unwrapped
    assert oc.api_key == "openai-secret"
    assert oc.api_version == "2025-01-01"
    assert oc.max_retries == 3
