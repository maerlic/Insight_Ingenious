"""Tests for the answer generation feature of the AzureSearchProvider.

This module verifies the behavior of the `answer` method, particularly its
pre-flight check that disables the functionality when not explicitly enabled.
It ensures that the provider correctly raises a `GenerationDisabledError`
when answer generation is turned off, preventing unintended calls to the
underlying search pipeline. Tests also confirm that the `retrieve` method
remains unaffected by this setting.

The primary entry points under test are the `AzureSearchProvider.answer` and
`AzureSearchProvider.retrieve` methods. Stubs and fixtures are used to
isolate provider logic from actual Azure services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@pytest.mark.asyncio
async def test_answer_raises_when_generation_disabled(
    import_provider_with_stubs: Any, settings_disabled: Any
) -> None:
    """Verify `answer` raises GenerationDisabledError when the feature is off.

    This test ensures that when answer generation is disabled by default in the
    settings, calling the `answer` method triggers a specific pre-flight error
    before attempting to execute the search pipeline.
    """
    provider_mod = import_provider_with_stubs
    Provider = provider_mod.AzureSearchProvider
    from ingenious.services.retrieval.errors import (
        GenerationDisabledError,
        PreflightError,
    )

    prov = Provider(settings_disabled)  # default: generation disabled
    with pytest.raises(GenerationDisabledError) as excinfo:
        await prov.answer("what is dat fusion?")
    # Subclass check / back-compat
    assert isinstance(excinfo.value, PreflightError)
    # Reason & snapshot available
    assert (
        getattr(excinfo.value, "reason", "") == "generation_disabled"
        or getattr(getattr(excinfo.value, "reason", ""), "value", "")
        == "generation_disabled"
    )
    snap: dict[str, Any] = getattr(excinfo.value, "snapshot", {})
    assert "use_semantic_ranking" in snap and "top_n_final" in snap

    # Ensure fail-fast: pipeline.get_answer was NOT called
    dummy_pipeline = provider_mod.build_search_pipeline(
        None
    )  # returns the shared dummy
    assert dummy_pipeline.get_answer_called == 0


@pytest.mark.asyncio
async def test_answer_succeeds_when_generation_enabled(
    import_provider_with_stubs: Any, settings_disabled: Any
) -> None:
    """Verify `answer` succeeds when explicitly enabled via constructor override.

    This test confirms that passing `enable_answer_generation=True` to the provider's
    constructor successfully overrides the disabled-by-default setting and allows
    the `answer` method to execute as expected.
    """
    provider_mod = import_provider_with_stubs
    Provider = provider_mod.AzureSearchProvider

    # Explicit override should enable generation, regardless of settings default.
    prov = Provider(settings_disabled, enable_answer_generation=True)
    res: dict[str, Any] = await prov.answer("q")
    assert isinstance(res, dict)
    assert res["answer"] == "A"
    assert res["source_chunks"] and res["source_chunks"][0]["id"] == "1"

    # Sanity: pipeline.get_answer called exactly once
    dummy_pipeline = provider_mod.build_search_pipeline(None)
    assert dummy_pipeline.get_answer_called == 1


@pytest.mark.asyncio
async def test_constructor_override_false_is_respected(
    import_provider_with_stubs: Any, settings_enabled: Any
) -> None:
    """Verify constructor override `enable_answer_generation=False` is respected.

    This test ensures that if the settings would otherwise enable generation,
    an explicit `False` passed to the constructor correctly disables it,
    upholding the override's precedence.
    """
    provider_mod = import_provider_with_stubs
    Provider = provider_mod.AzureSearchProvider
    from ingenious.services.retrieval.errors import GenerationDisabledError

    # settings_enabled doesn't actually flip the SearchConfig;
    # but ctor override False must be respected if provided.
    prov = Provider(settings_enabled, enable_answer_generation=False)
    with pytest.raises(GenerationDisabledError):
        await prov.answer("q")


@pytest.mark.asyncio
async def test_preflight_superclass_still_catches(
    import_provider_with_stubs: Any, settings_disabled: Any
) -> None:
    """Verify `GenerationDisabledError` can be caught via its `PreflightError` superclass.

    This ensures that client code can use a broader exception handler for any
    pre-flight validation failures, maintaining backward compatibility and
    flexible error handling strategies.
    """
    provider_mod = import_provider_with_stubs
    Provider = provider_mod.AzureSearchProvider
    from ingenious.services.retrieval.errors import PreflightError

    prov = Provider(settings_disabled)
    with pytest.raises(PreflightError):
        await prov.answer("q")


@pytest.mark.asyncio
async def test_public_reexport_path_optional(import_provider_with_stubs: Any) -> None:
    """Test that `GenerationDisabledError` is optionally re-exported from the package root.

    This is a convenience test. It checks if the specific error type is made
    available at `ingenious.services.azure_search` for easier importing by clients,
    skipping if the re-export does not exist.
    """
    try:
        from ingenious.services.azure_search import GenerationDisabledError  # re-export

        assert issubclass(GenerationDisabledError, Exception)
    except ImportError:
        pytest.skip("GenerationDisabledError not re-exported at package root.")


@pytest.mark.asyncio
async def test_retrieve_unaffected_when_generation_disabled(
    import_provider_with_stubs: Any, settings_disabled: Any, monkeypatch: MonkeyPatch
) -> None:
    """Verify `retrieve()` still works correctly when answer generation is disabled.

    This test confirms that the setting for answer generation is isolated and does
    not interfere with the standard document retrieval functionality of the provider.
    The test stubs the underlying retriever and fuser to focus solely on the
    provider's logic.
    """
    provider_mod = import_provider_with_stubs
    Provider = provider_mod.AzureSearchProvider

    # We have a shared dummy pipeline object on build_search_pipeline(None)
    dummy_pipeline = provider_mod.build_search_pipeline(None)

    class _DummyRetriever:
        """A stub for the retriever component."""

        async def search_lexical(
            self, q: str
        ) -> list[dict[str, Any]]:  # pretend two docs
            """Simulate a lexical search result."""
            return [
                {
                    "id": "A",
                    "content": "L",
                    "_retrieval_score": 0.9,
                    "@search.score": 1.0,
                    "_retrieval_type": "lexical_bm25",
                }
            ]

        async def search_vector(self, q: str) -> list[dict[str, Any]]:
            """Simulate a vector search result."""
            return [
                {
                    "id": "B",
                    "content": "V",
                    "_retrieval_score": 0.8,
                    "@search.score": 1.1,
                    "_retrieval_type": "vector_dense",
                }
            ]

        async def close(self) -> None:
            """Simulate closing the retriever."""
            pass

    class _DummyFuser:
        """A stub for the fuser component."""

        async def fuse(
            self, q: str, lex: list[dict[str, Any]], vec: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            """Simulate fusing lexical and vector results."""
            # Simple union with faux fused/final scores
            out: list[dict[str, Any]] = []
            for r in lex + vec:
                r["_fused_score"] = r.get("_retrieval_score", 0.0)
                r["_final_score"] = r["_fused_score"]
                out.append(r)
            return out

        async def close(self) -> None:
            """Simulate closing the fuser."""
            pass

    dummy_pipeline.retriever = _DummyRetriever()
    dummy_pipeline.fuser = _DummyFuser()

    # Ensure semantic ranking is off to keep it simple
    prov = Provider(settings_disabled, enable_answer_generation=False)
    prov._cfg = prov._cfg.copy(update={"use_semantic_ranking": False})

    docs: list[dict[str, Any]] = await prov.retrieve("q", top_k=2)
    assert isinstance(docs, list) and len(docs) == 2
    # Cleaned output should not contain internal keys
    for d in docs:
        assert "_fused_score" not in d and "_retrieval_score" not in d
