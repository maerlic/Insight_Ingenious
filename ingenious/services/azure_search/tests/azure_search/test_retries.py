# ingenious/services/azure_search/tests/azure_search/test_retries.py

import asyncio
from types import SimpleNamespace

import pytest

# Keep this import path as in your project
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever


class _AsyncResults:
    """Mimic the async iterator returned by SearchClient.search()."""

    def __init__(self, docs):
        self._docs = docs
        self._it = None

    def __aiter__(self):
        self._it = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class FlakySearchClient:
    """
    Simulates an Azure Search client with internal retry policy:
    two 429 "failures" then success — all inside a single .search() call.
    """

    def __init__(self):
        self.calls = 0  # how many times retriever called .search()
        self.attempts = 0  # internal retry attempts within the client

    async def search(self, *args, **kwargs):
        self.calls += 1
        # Simulate the SDK's internal retry loop: 2 failures then success
        for _ in range(3):
            self.attempts += 1
            if self.attempts <= 2:
                # pretend we saw a 429 and the client will retry internally
                # (no need to raise; the SDK would catch and retry)
                await asyncio.sleep(0)
                continue
            # success on third attempt
            return _AsyncResults([{"id": "1", "content": "ok", "@search.score": 2.0}])


class DummyEmbeddingsClient:
    """Minimal embeddings client (not used by lexical test, but required by ctor)."""

    def __init__(self):
        self.embeddings = self

    async def create(self, *args, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0])])


@pytest.mark.asyncio
async def test_retry_on_429_then_success():
    """
    Validates we succeed through the normal call path when the underlying
    client handles transient 429s via its own retry policy.
    """
    cfg = SimpleNamespace(
        top_k_retrieval=1,
        embedding_deployment_name="unused-here",
    )

    client = FlakySearchClient()
    r = AzureSearchRetriever(
        config=cfg, search_client=client, embedding_client=DummyEmbeddingsClient()
    )

    # NOTE: search_lexical only takes (query), top_k is read from config
    docs = await r.search_lexical("hello")

    # Retriever should call the client's .search() exactly once,
    # while the client itself performed 3 internal attempts.
    assert client.calls == 1, "Retriever should invoke SearchClient.search once"
    assert client.attempts == 3, "Expected 2 internal retries (3 total attempts)"
    assert isinstance(docs, list) and docs and docs[0]["content"] == "ok"


class FlakyEmbeddingsClient:
    """
    Async embeddings client that simulates internal retry:
    two transient failures then success within a single .create() call.
    """

    def __init__(self):
        self.attempts = 0
        self.embeddings = self  # expose .create via 'embeddings' attribute

    async def create(self, *args, **kwargs):
        # Simulate internal retry loop
        for _ in range(3):
            self.attempts += 1
            if self.attempts <= 2:
                await asyncio.sleep(0)
                continue
            # success shape: .data[0].embedding
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class NoopSearchClient:
    """A search client we won't actually use for this test."""

    async def search(self, *args, **kwargs):
        return _AsyncResults([])


@pytest.mark.asyncio
async def test_vector_embed_429_fallback_or_retry():
    """
    Embedding call should succeed even if the client had to retry internally.
    Validates we propagate our retry-configured client through the call path.
    """
    cfg = SimpleNamespace(
        top_k_retrieval=1,
        embedding_deployment_name="my-embedding-deployment",
    )

    aoai = FlakyEmbeddingsClient()
    r = AzureSearchRetriever(
        config=cfg, search_client=NoopSearchClient(), embedding_client=aoai
    )

    vec = await r._generate_embedding("hello")  # protected method ok for unit scope

    assert isinstance(vec, (list, tuple)), "Expected embedding vector returned"
    assert aoai.attempts == 3, (
        "Expected 2 internal retries (3 total attempts) for embeddings.create"
    )
