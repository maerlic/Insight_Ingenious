import pytest
import typer

from ingenious.services.azure_search import cli


def test_cli_semantic_enabled_without_name_exits_1(config, monkeypatch):
    """
    When the pipeline factory (build_search_pipeline) raises ValueError,
    _run_search_pipeline should exit(1) instead of silently 'succeeding'.
    """

    def raise_value_error(*_a, **_k):
        raise ValueError(
            "semantic ranking is enabled but semantic config name is missing"
        )

    # Make the factory blow up; no need to construct a special config since we force the error.
    monkeypatch.setattr(cli, "build_search_pipeline", raise_value_error, raising=False)

    with pytest.raises(typer.Exit) as ei:
        # NOTE: correct signature is (config, query, verbose)
        cli._run_search_pipeline(config, "hello world", False)

    assert ei.value.exit_code == 1
