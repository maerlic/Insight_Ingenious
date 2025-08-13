# ingenious/services/azure_search/tests/azure_search/test_concurrency_cancellation.py
import asyncio
from types import SimpleNamespace

import pytest

# keep this import — we're exercising the real retrieve() implementation
from ingenious.services.azure_search.provider import AzureSearchProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_branch", ["vector", "lexical"])
async def test_l1_other_branch_cancelled_on_failure(failing_branch):
    """
    Ensure we don't leak work when one L1 branch fails: the sibling must be cancelled.
    One branch raises fast, the other sleeps (and should observe CancelledError).
    """
    cancelled = asyncio.Event()

    async def slow_sleeping_search(_q: str):
        try:
            # long sleep so gather() must cancel us for the test to complete
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def fast_failing_search(_q: str):
        raise TimeoutError("simulated L1 failure")

    # build provider WITHOUT running __init__, then inject only what retrieve() touches
    provider = object.__new__(AzureSearchProvider)

    # minimal cfg so any code path that might touch it won't explode
    provider._cfg = SimpleNamespace(
        use_semantic_ranking=False,
        vector_field="content_vector",
        id_field="id",
        semantic_configuration_name=None,
    )

    # stub retriever with the exact method names retrieve() calls
    retriever = SimpleNamespace()
    if failing_branch == "vector":
        retriever.search_vector = fast_failing_search
        retriever.search_lexical = slow_sleeping_search
    else:
        retriever.search_vector = slow_sleeping_search
        retriever.search_lexical = fast_failing_search

    async def _should_not_run(*_a, **_k):
        pytest.fail("fuser.fuse should not be reached when an L1 branch fails")

    # inject a pipeline that matches the provider's interface
    provider._pipeline = SimpleNamespace(
        retriever=retriever,
        fuser=SimpleNamespace(fuse=_should_not_run),
    )

    # provider.retrieve should propagate the failure and cancel the sibling
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.retrieve("q"), timeout=2.0)

    # prove the sleeping branch actually got cancelled (not just left running)
    await asyncio.wait_for(cancelled.wait(), timeout=0.5)
