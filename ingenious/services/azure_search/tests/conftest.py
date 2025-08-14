# -- coding: utf-8 --

"""
COVERAGE PLAN

    Provide common fixtures, dummies, and precise monkeypatching to keep tests offline.
    Replace Azure/OpenAI clients with AsyncMock instances returned by our own factories.
    Patch QueryType/VectorizedQuery used by retrieval module to avoid azure SDK import needs.
    Build a canonical SearchConfig with dummy secrets and toggle variant without semantic ranking.

    Helpers:
        AsyncIter for async iteration in semantic reranking paths.
        make_search_client / make_async_openai_client monkeypatch targets.

    Ensure components/pipeline/cli import paths work whether package is editable or installed.
"""

# ---- EARLY Azure SDK stubs: must run at import-time, not as a fixture ----
import sys
import types
from types import SimpleNamespace
from typing import List

# tests/conftest.py
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

# Public model under test
from ingenious.services.azure_search.config import DEFAULT_DAT_PROMPT, SearchConfig


def _install_azure_stubs() -> None:
    # Helpers that *augment* existing modules if present
    def _ensure_pkg(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
        else:
            if not hasattr(mod, "__path__"):
                mod.__path__ = []
        return mod

    def _ensure_mod(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return mod

    # Root namespaces
    _ensure_pkg("azure")
    _ensure_pkg("azure.core")

    # ---- azure.core.credentials ----
    core_credentials = _ensure_mod("azure.core.credentials")
    if not hasattr(core_credentials, "AzureKeyCredential"):

        class AzureKeyCredential:
            def __init__(self, key: str):
                self.key = key

        core_credentials.AzureKeyCredential = AzureKeyCredential

    # ---- azure.core.rest ----
    core_rest = _ensure_mod("azure.core.rest")
    if not hasattr(core_rest, "HttpResponse"):

        class HttpResponse:
            def __init__(
                self, *, status_code=200, reason=None, headers=None, text=None
            ):
                self.status_code = status_code
                self.reason = reason or ""
                self.headers = headers or {}
                self._text = text or ""

            def text(self):  # minimal compat
                return self._text

        core_rest.HttpResponse = HttpResponse

    # ---- azure.core.exceptions ----
    core_exceptions = _ensure_mod("azure.core.exceptions")
    if not hasattr(core_exceptions, "AzureError"):

        class AzureError(Exception):
            def __init__(self, message=None, **kwargs):
                super().__init__(message or "")
                self.message = message or ""
                self.kwargs = kwargs

        class HttpResponseError(AzureError):
            def __init__(
                self,
                message=None,
                *,
                response=None,
                error=None,
                status_code=None,
                **kwargs,
            ):
                super().__init__(message, **kwargs)
                self.response = response
                self.error = error
                self.status_code = (
                    status_code
                    if status_code is not None
                    else getattr(response, "status_code", None)
                )

            def __str__(self):
                sc = self.status_code
                return (
                    f"HttpResponseError({sc}): {self.message}"
                    if sc is not None
                    else f"HttpResponseError: {self.message}"
                )

        class ClientAuthenticationError(HttpResponseError): ...

        class ResourceNotFoundError(HttpResponseError): ...

        class ServiceRequestError(AzureError): ...

        class ServiceResponseError(AzureError): ...

        core_exceptions.AzureError = AzureError
        core_exceptions.HttpResponseError = HttpResponseError
        core_exceptions.ClientAuthenticationError = ClientAuthenticationError
        core_exceptions.ResourceNotFoundError = ResourceNotFoundError
        core_exceptions.ServiceRequestError = ServiceRequestError
        core_exceptions.ServiceResponseError = ServiceResponseError

    # ---- azure.identity (sync) ----
    identity = _ensure_mod("azure.identity")
    if not hasattr(identity, "DefaultAzureCredential"):

        class _BaseCredential:
            def __init__(self, *a, **k):
                pass

            def get_token(self, *scopes, **kwargs):
                return types.SimpleNamespace(token="fake-token", expires_on=0)

        class DefaultAzureCredential(_BaseCredential): ...

        class AzureDeveloperCliCredential(_BaseCredential): ...

        class ManagedIdentityCredential(_BaseCredential): ...

        class VisualStudioCodeCredential(_BaseCredential): ...

        class ClientSecretCredential(_BaseCredential):
            def __init__(self, tenant_id, client_id, client_secret, **kwargs):
                super().__init__(tenant_id, client_id, client_secret, **kwargs)

        class ClientCertificateCredential(_BaseCredential):
            def __init__(self, tenant_id, client_id, certificate_path=None, **kwargs):
                super().__init__(tenant_id, client_id, certificate_path, **kwargs)

        class EnvironmentCredential(_BaseCredential): ...

        class InteractiveBrowserCredential(_BaseCredential): ...

        class DeviceCodeCredential(_BaseCredential): ...

        class ChainedTokenCredential(_BaseCredential):
            def __init__(self, *credentials):
                self._creds = credentials

            def get_token(self, *scopes, **kwargs):
                for c in self._creds:
                    tok = getattr(c, "get_token", lambda *a, **k: None)(
                        *scopes, **kwargs
                    )
                    if tok:
                        return tok
                return super().get_token(*scopes, **kwargs)

        def get_bearer_token_provider(credential, scope):
            def _provider(*a, **k):
                return "fake-token"

            return _provider

        identity.DefaultAzureCredential = DefaultAzureCredential
        identity.AzureDeveloperCliCredential = AzureDeveloperCliCredential
        identity.ManagedIdentityCredential = ManagedIdentityCredential
        identity.VisualStudioCodeCredential = VisualStudioCodeCredential
        identity.ClientSecretCredential = ClientSecretCredential
        identity.ClientCertificateCredential = ClientCertificateCredential
        identity.EnvironmentCredential = EnvironmentCredential
        identity.InteractiveBrowserCredential = InteractiveBrowserCredential
        identity.DeviceCodeCredential = DeviceCodeCredential
        identity.ChainedTokenCredential = ChainedTokenCredential
        identity.get_bearer_token_provider = get_bearer_token_provider

    # ---- azure.identity.aio (async mirror) ----
    identity_aio = _ensure_mod("azure.identity.aio")
    for name in (
        "DefaultAzureCredential",
        "AzureDeveloperCliCredential",
        "ManagedIdentityCredential",
        "VisualStudioCodeCredential",
        "ClientSecretCredential",
        "ClientCertificateCredential",
        "EnvironmentCredential",
        "InteractiveBrowserCredential",
        "DeviceCodeCredential",
        "ChainedTokenCredential",
        "get_bearer_token_provider",
    ):
        if not hasattr(identity_aio, name):
            setattr(identity_aio, name, getattr(identity, name))

    # ---- azure.keyvault.secrets (sync + aio) ----
    _ensure_pkg("azure.keyvault")
    secrets_mod = _ensure_mod("azure.keyvault.secrets")
    if not hasattr(secrets_mod, "SecretClient"):

        class _KeyVaultSecret:
            def __init__(self, name, value):
                self.name = name
                self.value = value
                # minimal properties bag seen in some code paths
                self.properties = types.SimpleNamespace(enabled=True)

        class SecretClient:
            """Minimal Key Vault Secrets client used by config code in tests."""

            def __init__(self, vault_url, credential, **kwargs):
                self.vault_url = vault_url
                self.credential = credential
                self._store = {}  # simple in-memory secret store

            def get_secret(self, name, **kwargs):
                if name in self._store:
                    return self._store[name]
                from azure.core.exceptions import ResourceNotFoundError

                raise ResourceNotFoundError(f"Secret '{name}' not found")

            def set_secret(self, name, value, **kwargs):
                s = _KeyVaultSecret(name, value)
                self._store[name] = s
                return s

        secrets_mod.SecretClient = SecretClient
        secrets_mod.KeyVaultSecret = _KeyVaultSecret

    secrets_aio = _ensure_mod("azure.keyvault.secrets.aio")
    if not hasattr(secrets_aio, "SecretClient"):

        class _AioSecretClient:
            def __init__(self, vault_url, credential, **kwargs):
                # reuse sync behavior
                self._sync = secrets_mod.SecretClient(vault_url, credential, **kwargs)

            async def get_secret(self, name, **kwargs):
                return self._sync.get_secret(name, **kwargs)

            async def set_secret(self, name, value, **kwargs):
                return self._sync.set_secret(name, value, **kwargs)

            async def close(self):
                pass

        secrets_aio.SecretClient = _AioSecretClient

    # ---- azure.search.documents.* ----
    _ensure_pkg("azure.search")
    _ensure_pkg("azure.search.documents")

    docs_aio = _ensure_mod("azure.search.documents.aio")
    if not hasattr(docs_aio, "SearchClient"):

        class _SearchClient:
            async def search(self, *args, **kwargs):
                async def _aiter():
                    if False:  # pragma: no cover
                        yield {}

                return _aiter()

            async def close(self):
                pass

        docs_aio.SearchClient = _SearchClient

    docs_models = _ensure_mod("azure.search.documents.models")
    if not hasattr(docs_models, "QueryType"):

        class _QueryType:
            SIMPLE = 1
            SEMANTIC = 2

        class _VectorizedQuery:
            def __init__(self, *, vector, k_nearest_neighbors, fields, exhaustive=True):
                self.vector = vector
                self.k_nearest_neighbors = k_nearest_neighbors
                self.fields = fields
                self.exhaustive = exhaustive

        docs_models.QueryType = _QueryType
        docs_models.VectorizedQuery = _VectorizedQuery


_install_azure_stubs()


# -------------------------------------------------------------------------
def _ensure_stub_module(name: str, attrs: dict):
    """Make a stub importable module so patches on it won't fail if it's not installed."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod


# Allow patch('chromadb.PersistentClient', ...) even if chromadb isn't installed.
_ensure_stub_module("chromadb", {"PersistentClient": MagicMock()})


@pytest.fixture(autouse=True, scope="session")
def _preseed_tool_factory_patch_targets():
    """
    Some tests patch attributes on tool_factory that aren't exported by prod code.
    Precreate harmless placeholders so unittest.mock.patch(...) can replace them.
    """
    import importlib

    tf = importlib.import_module(
        "ingenious.services.chat_services.multi_agent.tool_factory"
    )

    if not hasattr(tf, "AzureSearchProvider"):
        tf.AzureSearchProvider = object  # type: ignore[attr-defined]

    if not hasattr(tf, "get_config"):

        def _placeholder_get_config():
            raise RuntimeError("get_config placeholder: tests must patch this symbol.")

        tf.get_config = _placeholder_get_config  # type: ignore[attr-defined]


@pytest.fixture
def mock_search_config(config) -> SearchConfig:
    """Alias the suite's existing 'config' fixture to the name these tests expect."""
    return config


@pytest.fixture
def mock_ingenious_settings():
    """Minimal settings shape used by tests."""
    svc = SimpleNamespace(
        endpoint="https://unit.search.windows.net",
        key="unit-key",
        index_name="unit-index",
    )
    return SimpleNamespace(azure_search_services=[svc])


@pytest.fixture
def mock_async_openai_client():
    """AsyncOpenAI-like stub path used by fusion tests."""
    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat_completions = SimpleNamespace()  # tolerate alt path if needed
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = AsyncMock()
    return client


# --- Async iterator helper ---------------------------------------------------
class AsyncIter:
    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


# --- Dummy Azure SDK model stand-ins -----------------------------------------


class DummyQueryType:
    SIMPLE = "simple"
    SEMANTIC = "semantic"


class DummyVectorizedQuery:
    def __init__(
        self,
        *,
        vector: List[float],
        k_nearest_neighbors: int,
        fields: str,
        exhaustive: bool = True,
    ):
        self.vector = vector
        self.k_nearest_neighbors = k_nearest_neighbors
        self.fields = fields
        self.exhaustive = exhaustive


# --- Core config fixtures -----------------------------------------------------


@pytest.fixture
def config() -> SearchConfig:
    """Valid SearchConfig for most tests."""
    return SearchConfig(
        search_endpoint="https://unit-search.windows.net",
        search_key=SecretStr("search_key"),
        search_index_name="unit-index",
        semantic_configuration_name="sem-config",
        openai_endpoint="https://unit-openai.azure.com",
        openai_key=SecretStr("openai_key"),
        openai_version="2024-02-01",
        embedding_deployment_name="embed-deploy",
        generation_deployment_name="chat-deploy",
        top_k_retrieval=10,
        use_semantic_ranking=True,
        top_n_final=3,
        id_field="id",
        content_field="content",
        vector_field="vector",
        dat_prompt=DEFAULT_DAT_PROMPT,
    )


@pytest.fixture
def config_no_semantic(config: SearchConfig) -> SearchConfig:
    """Variant with semantic ranking disabled."""
    data = config.model_dump(exclude={"search_key", "openai_key"})
    data["use_semantic_ranking"] = False
    data["semantic_configuration_name"] = None
    data["search_key"] = config.search_key
    data["openai_key"] = config.openai_key
    return SearchConfig(**data)


# --- Global monkeypatches for external deps ----------------------------------


@pytest.fixture(autouse=True)
def patch_external_sdks(monkeypatch):
    """
    - Patch azure.search.documents.models symbols referenced directly by modules.
    - Patch client factory functions used by components/pipeline.
    - Provide OpenAI-like Async client with embeddings/chat APIs.
    """
    # Patch model symbols where modules import them
    monkeypatch.setitem(globals(), "DummyQueryType", DummyQueryType)
    monkeypatch.setitem(globals(), "DummyVectorizedQuery", DummyVectorizedQuery)

    # Retrieval module uses these names at import time
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "0")  # make asyncio errors clearer

    # Patch azure.search.documents.models.QueryType & VectorizedQuery to our dummies
    for target in [
        "ingenious.services.azure_search.components.retrieval.QueryType",
        "ingenious.services.azure_search.components.pipeline.QueryType",
    ]:
        try:
            monkeypatch.setattr(target, DummyQueryType, raising=False)
        except Exception:
            pass
    try:
        monkeypatch.setattr(
            "ingenious.services.azure_search.components.retrieval.VectorizedQuery",
            DummyVectorizedQuery,
            raising=False,
        )
    except Exception:
        pass

    # Create shared async OpenAI mock with embeddings & chat
    openai_client = AsyncMock(name="AsyncAzureOpenAI")
    openai_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.01] * 3)])
    )
    openai_client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="3 4"))]
        )
    )

    # Create shared search client mock with .search() async iterator and .close()
    search_client = AsyncMock(name="SearchClient")
    search_client.search = AsyncMock(return_value=AsyncIter([]))
    search_client.close = AsyncMock()

    # Patch factory functions to return our shared clients
    monkeypatch.setattr(
        "ingenious.services.azure_search.client_init.make_async_openai_client",
        lambda cfg: openai_client,
        raising=False,
    )
    monkeypatch.setattr(
        "ingenious.services.azure_search.client_init.make_search_client",
        lambda cfg: search_client,
        raising=False,
    )

    # Components import factories dynamically; also patch potential direct paths
    for mod in [
        "ingenious.services.azure_search.components.retrieval",
        "ingenious.services.azure_search.components.fusion",
        "ingenious.services.azure_search.components.generation",
        "ingenious.services.azure_search.pipeline",
    ]:
        try:
            monkeypatch.setattr(
                f"{mod}.make_async_openai_client",
                lambda cfg: openai_client,
                raising=False,
            )
        except Exception:
            pass
        try:
            monkeypatch.setattr(
                f"{mod}.make_search_client", lambda cfg: search_client, raising=False
            )
        except Exception:
            pass

    # Expose to tests that need them
    yield


# Utility fixtures re-exposed for tests that want them


@pytest.fixture
def async_iter():  # noqa: D401
    """Return AsyncIter helper class."""
    return AsyncIter
