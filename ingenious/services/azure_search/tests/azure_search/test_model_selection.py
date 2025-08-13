# tests/azure_search/test_model_selection.py

from types import SimpleNamespace

import pytest

# ⬇️ ADJUST THESE IMPORTS to match where your builders live
from ingenious.services.azure_search.builders import (
    ConfigError,
    _pick_models,
    build_search_config_from_settings,
)


def _settings(**overrides):
    """
    Tiny helper that mimics the shape your builders expect.
    If your project already exposes a fixture/helper, prefer that instead.
    """
    # Minimal model items: provider, role, deployment, endpoint/key, api_version
    models = overrides.pop(
        "models",
        [
            # Chat model with deployment
            SimpleNamespace(
                provider="azure_openai",
                role="chat",
                endpoint="https://aoai.example.com",
                key="x",
                api_version="2024-02-15-preview",
                deployment="gpt-4o",
                model="gpt-4o",
            ),
            # Embedding model WITHOUT deployment (this is the case we want to fail)
            SimpleNamespace(
                provider="azure_openai",
                role="embedding",
                endpoint="https://aoai.example.com",
                key="x",
                api_version="2024-02-15-preview",
                deployment="",  # ❌ missing
                model="text-embedding-3-large",
            ),
        ],
    )

    # Minimal azure search service entry
    azure_search_services = overrides.pop(
        "azure_search_services",
        [
            SimpleNamespace(
                endpoint="https://acct.search.windows.net",
                key="x",
                index_name="idx",
                semantic_ranking=True,
                semantic_configuration="my-semantic",
            )
        ],
    )

    return SimpleNamespace(
        models=models, azure_search_services=azure_search_services, **overrides
    )


def test_pick_models_requires_embedding_deployment():
    """
    Ensures _pick_models fails when an embedding model is present but its deployment is empty.
    """
    s = _settings()
    with pytest.raises(
        ValueError
    ):  # The builders typically raise ValueError for selection faults
        _pick_models(s)


@pytest.mark.parametrize(
    "embed_dep, chat_dep, should_raise",
    [
        ("emb-001", "gpt-4o", False),  # different deployments — OK
        (
            "shared-dep",
            "shared-dep",
            True,
        ),  # same deployments — should raise if guard is enabled
    ],
)
def test_builder_rejects_same_deployments_for_embed_and_chat(
    embed_dep, chat_dep, should_raise
):
    """
    Optional policy: enforce that embedding and chat deployments differ.
    If you haven't added the guard yet, mark this test xfail (see below).
    """
    s = _settings(
        models=[
            SimpleNamespace(
                provider="azure_openai",
                role="embedding",
                endpoint="https://aoai.example.com",
                key="x",
                api_version="2024-02-15-preview",
                deployment=embed_dep,
                model="text-embedding-3-large",
            ),
            SimpleNamespace(
                provider="azure_openai",
                role="chat",
                endpoint="https://aoai.example.com",
                key="x",
                api_version="2024-02-15-preview",
                deployment=chat_dep,
                model="gpt-4o",
            ),
        ]
    )

    if should_raise:
        with pytest.raises(ConfigError):
            build_search_config_from_settings(s)
    else:
        cfg = build_search_config_from_settings(s)
        # Sanity: the builder still returns a config object if deployments differ
        assert hasattr(cfg, "openai"), (
            "Expected a SearchConfig-like object with .openai fields"
        )
