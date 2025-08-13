# tests/services/azure_search/test_numeric_knobs_validation.py
import pytest

try:
    # Prefer validating through the builder (recommended change).
    from ingenious.services.azure_search.builders import (
        build_search_config_from_settings as build_cfg,
    )

    USING_BUILDER = True
except Exception:  # pragma: no cover - fall back to model-level validation if needed
    USING_BUILDER = False

# Fallback: if you instead add constrained ints on the model, this path will cover it.
if not USING_BUILDER:
    # Adjust this import if your config model lives elsewhere:
    from ingenious.services.azure_search.config import SearchConfig  # type: ignore


@pytest.mark.parametrize(
    "top_k_retrieval, top_n_final",
    [
        (0, 10),  # top_k == 0
        (-1, 10),  # top_k < 0
        (10, 0),  # top_n == 0
        (10, -5),  # top_n < 0
        (0, 0),  # both invalid
    ],
)
def test_builder_rejects_non_positive_topk_topn(
    monkeypatch, top_k_retrieval, top_n_final
):
    """
    Defensive validation: reject non-positive values for top_k_retrieval/top_n_final.

    This test passes whether you enforce the rule inside:
      - build_search_config_from_settings (raising ValueError), OR
      - the SearchConfig pydantic model (raising ValidationError).
    """

    if USING_BUILDER:
        # Build a minimal settings stub for the builder.
        # NOTE: Adjust attribute names if your builder expects different ones.
        from types import SimpleNamespace

        # Minimal Azure Search service stanza the builder reads from settings.azure_search_services[0]
        service = SimpleNamespace(
            endpoint="https://example.search.windows.net",
            api_key="test-key",
            index_name="test-index",
            use_semantic_ranking=False,
            semantic_configuration_name=None,
            # Add the top_k and top_n values here for the test
            top_k_retrieval=top_k_retrieval,
            top_n_final=top_n_final,
        )

        # Provide settings with the service
        settings = SimpleNamespace(
            azure_search_services=[service],
        )

        # Mock _pick_models to return a tuple of 5 values as expected
        def mock_pick_models(_settings):
            return (
                "https://aoai.local",  # openai_endpoint
                "sk-test",  # openai_key
                "2024-05-01-preview",  # openai_version
                "embedding-deploy",  # embedding_deployment
                "chat-deploy",  # generation_deployment
            )

        import ingenious.services.azure_search.builders as builders

        monkeypatch.setattr(builders, "_pick_models", mock_pick_models)

        with pytest.raises(ValueError):
            build_cfg(settings)

    else:
        # Model-level validation path (if you chose to enforce via constrained ints on SearchConfig)
        from pydantic import ValidationError

        # NOTE: Adjust required fields according to your SearchConfig definition.
        with pytest.raises(ValidationError):
            SearchConfig(
                # Azure Search bits
                search_endpoint="https://example.search.windows.net",
                search_api_key="test-key",
                index_name="test-index",
                # AOAI bits (if your model carries them)
                openai_endpoint="https://aoai.local",
                openai_api_key="sk-test",
                openai_api_version="2024-05-01-preview",
                embedding_deployment="embedding-deploy",
                chat_deployment="chat-deploy",
                # Knobs under test
                top_k_retrieval=top_k_retrieval,
                top_n_final=top_n_final,
            )
