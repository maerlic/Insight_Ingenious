"""
Tests concurrency cancellation logic for the Azure Search provider.

This module ensures that when performing concurrent L1 searches (vector and
lexical), a failure in one search branch correctly triggers the cancellation
of the other. This prevents orphaned, long-running tasks and resource leaks,
which is critical for system stability under partial failures. The main entry
point exercised is `AzureSearchProvider.retrieve()`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

# keep this import — we're exercising the real retrieve() implementation
from ingenious.services.azure_search.provider import AzureSearchProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_branch", ["vector", "lexical"])
async def test_l1_other_branch_cancelled_on_failure(failing_branch: str) -> None:
    """
    Ensure a failing L1 search branch cancels its sibling to prevent leaked work.

    This test validates the `asyncio.gather` exception handling within the
    `retrieve` method. It simulates a scenario where one search (vector or
    lexical) fails instantly, while the other is a long-running task. The test
    asserts that the long-running task is cancelled promptly, rather than being
    allowed to complete or run indefinitely.
    """
    cancelled: asyncio.Event = asyncio.Event()

    async def slow_sleeping_search(_q: str) -> NoReturn:
        """
        Simulate a long-running search that should be cancelled.

        This coroutine sleeps for an extended period to mimic a slow network
        request. It's designed to be cancelled by the surrounding `gather`
        when a sibling task fails. Upon cancellation, it sets an event to
        signal that it was correctly terminated. It should never return normally.
        """
        try:
            # long sleep so gather() must cancel us for the test to complete
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        pytest.fail("slow_sleeping_search completed instead of being cancelled.")

    async def fast_failing_search(_q: str) -> NoReturn:
        """
        Simulate an immediate L1 search failure.

        This coroutine raises a `TimeoutError` instantly to represent a
        failed dependency, triggering the cancellation logic under test.
        """
        raise TimeoutError("simulated L1 failure")

    # build provider WITHOUT running __init__, then inject only what retrieve() touches
    provider: AzureSearchProvider = object.__new__(AzureSearchProvider)

    # minimal cfg so any code path that might touch it won't explode
    # mypy: The `_cfg` attribute is dynamically set on an un-initialized object.
    provider._cfg = SimpleNamespace(  # type: ignore[attr-defined]
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

    async def _should_not_run(*_a: Any, **_k: Any) -> NoReturn:
        """
        Fail the test if this function is ever called.

        This mock is injected as the `fuse` method to ensure that the fusion
        stage of the pipeline is not executed when an L1 retrieval branch fails.
        """
        pytest.fail("fuser.fuse should not be reached when an L1 branch fails")

    # inject a pipeline that matches the provider's interface
    # mypy: The `_pipeline` attribute is dynamically set on an un-initialized object.
    provider._pipeline = SimpleNamespace(  # type: ignore[attr-defined]
        retriever=retriever,
        fuser=SimpleNamespace(fuse=_should_not_run),
    )

    # provider.retrieve should propagate the failure and cancel the sibling
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.retrieve("q"), timeout=2.0)

    # prove the sleeping branch actually got cancelled (not just left running)
    await asyncio.wait_for(cancelled.wait(), timeout=0.5)
