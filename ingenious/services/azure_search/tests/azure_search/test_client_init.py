# -*- coding: utf-8 -*-
"""Unit tests for Azure client initialization logic.

This module contains unit tests for the `ingenious.services.azure_search.client_init`
module. It verifies that the client factory functions correctly construct and
configure Azure SDK clients based on a provided configuration object.

To avoid actual network calls or dependencies on the Azure SDKs during testing,
this module uses pytest's `monkeypatch` fixture to install dummy (mock) versions
of the required SDK modules and classes into `sys.modules` before the target
module is imported. This ensures that the client initialization logic binds to
our test doubles, allowing us to assert that they were called with the correct
parameters, such as unwrapped secrets and appropriate retry settings.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from ingenious.services.azure_search.config import SearchConfig

if TYPE_CHECKING:
    import pytest


def _install_dummy_sdk_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, type[Any]]:
    """Install minimal dummy SDK modules into sys.modules for testing.

    This function creates lightweight, fake versions of the Azure and OpenAI SDK
    classes that are dependencies for the `client_init` module. It then uses
    `monkeypatch` to insert these dummy modules into `sys.modules`. When the
    `client_init` module is subsequently imported (or reloaded), its `import`
    statements resolve to these dummies instead of the real SDKs. This allows
    for isolated unit testing without requiring the SDKs to be installed or
    configured.

    Args:
        monkeypatch: The pytest fixture for modifying modules, dicts, or os.environ.

    Returns:
        A dictionary mapping class names to the dummy class objects, which can
        be used in tests to make assertions (e.g., with `isinstance`).
    """

    # --- azure.core.credentials.AzureKeyCredential ---------------------------
    class DummyAzureKeyCredential:
        """A dummy replacement for AzureKeyCredential to capture the key."""

        def __init__(self, key: str) -> None:
            """Store the provided key for assertion."""
            self.key = key

    # --- azure.search.documents.aio.SearchClient -----------------------------
    class DummySearchClient:
        """A dummy replacement for SearchClient to capture constructor args."""

        def __init__(
            self,
            *,
            endpoint: str,
            index_name: str,
            credential: DummyAzureKeyCredential,
            **kwargs: Any,
        ) -> None:
            """Store endpoint, index, credential, and other args for assertion."""
            self.endpoint = endpoint
            self.index_name = index_name
            self.credential = credential
            # Accept any extra kwargs without requiring them
            self.extra_kwargs = kwargs

    # --- openai.AsyncAzureOpenAI --------------------------------------------
    class DummyAsyncAzureOpenAI:
        """A dummy replacement for AsyncAzureOpenAI to capture constructor args."""

        def __init__(
            self,
            *,
            azure_endpoint: str,
            api_key: str,
            api_version: str,
            max_retries: int,
        ) -> None:
            """Store endpoint, key, version, and retries for assertion."""
            self.azure_endpoint = azure_endpoint
            self.api_key = api_key
            self.api_version = api_version
            self.max_retries = max_retries

    # Build module objects and register in sys.modules
    azure = types.ModuleType("azure")
    core = types.ModuleType("azure.core")
    credentials = types.ModuleType("azure.core.credentials")
    credentials.AzureKeyCredential = DummyAzureKeyCredential  # type: ignore[attr-defined]

    search = types.ModuleType("azure.search")
    documents = types.ModuleType("azure.search.documents")
    aio = types.ModuleType("azure.search.documents.aio")
    aio.SearchClient = DummySearchClient  # type: ignore[attr-defined]

    openai_mod = types.ModuleType("openai")
    openai_mod.AsyncAzureOpenAI = DummyAsyncAzureOpenAI  # type: ignore[attr-defined]

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


def _reload_client_init_with_dummies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.ModuleType, dict[str, type[Any]]]:
    """Install dummy SDKs and then reload the client_init module.

    This is a test helper that orchestrates two key steps:
    1. It calls `_install_dummy_sdk_modules` to patch `sys.modules`.
    2. It then imports and reloads the `client_init` module, forcing it to
       bind to the newly installed dummy modules.

    Args:
        monkeypatch: The pytest fixture passed through to the installer.

    Returns:
        A tuple containing the reloaded `client_init` module and the dictionary
        of dummy classes for making assertions.
    """
    dummies = _install_dummy_sdk_modules(monkeypatch)
    # Now (re)load the module so it binds to our dummies
    import ingenious.services.azure_search.client_init as client_init

    client_init = importlib.reload(client_init)
    return client_init, dummies


def _make_cfg(**overrides: Any) -> SearchConfig:
    """Create a SearchConfig instance with default values for testing.

    This helper simplifies test setup by providing a valid `SearchConfig` object.
    Specific fields can be overridden by passing them as keyword arguments.

    Args:
        **overrides: Keyword arguments to override default config values.

    Returns:
        A configured `SearchConfig` instance.
    """
    data: dict[str, Any] = dict(
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
def test_make_search_client_uses_secretstr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify make_search_client unwraps the SecretStr for the credential key.

    This test ensures that the `make_search_client` factory function correctly
    constructs an `AzureKeyCredential` by extracting the raw string value from
    the `search_key` `SecretStr` in the configuration.
    """
    client_init, d = _reload_client_init_with_dummies(monkeypatch)

    cfg = _make_cfg()
    sc: Any = client_init.make_search_client(cfg)

    # Type and constructor args captured by our dummy
    assert isinstance(sc, d["SearchClient"])
    assert sc.endpoint == cfg.search_endpoint
    assert sc.index_name == cfg.search_index_name

    # SecretStr was unwrapped (not the SecretStr object itself)
    assert isinstance(sc.credential, d["AzureKeyCredential"])
    assert sc.credential.key == "search-secret"

    # No longer testing retry_policy since we removed it


def test_make_async_openai_client_maps_version_and_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify make_async_openai_client correctly passes config values.

    This test confirms that the factory function for the OpenAI client correctly
    maps the configuration fields (`openai_endpoint`, `openai_key`, `openai_version`)
    to the corresponding constructor arguments of the `AsyncAzureOpenAI` client.
    It also checks that it unwraps the `SecretStr` for the API key and hardcodes
    `max_retries` to a sensible default.
    """
    client_init, d = _reload_client_init_with_dummies(monkeypatch)

    cfg = _make_cfg(openai_version="2025-01-01")
    oc: Any = client_init.make_async_openai_client(cfg)

    assert isinstance(oc, d["AsyncAzureOpenAI"])
    assert oc.azure_endpoint == cfg.openai_endpoint
    # SecretStr was unwrapped
    assert oc.api_key == "openai-secret"
    assert oc.api_version == "2025-01-01"
    assert oc.max_retries == 3
