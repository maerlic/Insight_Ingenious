# ingenious/services/azure_search/tests/cli/test_azure_search_cli_behavior.py
import logging
from unittest.mock import patch

from typer.testing import CliRunner


def _base_env():
    # Minimal env so SearchConfig validation passes and we hit the CLI code paths
    return {
        "AZURE_SEARCH_ENDPOINT": "https://search.example.net",
        "AZURE_SEARCH_KEY": "search-key",
        "AZURE_SEARCH_INDEX_NAME": "my-index",
        "AZURE_OPENAI_ENDPOINT": "https://aoai.example.com",
        "AZURE_OPENAI_KEY": "openai-key",
    }


def _base_args(verbose: bool = False):
    args = [
        "azure-search",
        "run",
        "test query",
        "--embedding-deployment",
        "emb-deploy",
        "--generation-deployment",
        "gen-deploy",
        # leave out --semantic-config-name on purpose for the first test
        "--semantic-ranking",  # explicit ON (it's True by default, but be clear)
    ]
    if verbose:
        args.append("--verbose")
    return args


def test_cli_missing_semantic_name_exits_1():
    """
    With --semantic-ranking enabled and no --semantic-config-name, the pipeline
    builder should raise, and the CLI must exit(1) with an error panel message.
    We patch the CLI's build_search_pipeline shim to raise ValueError to assert
    the CLI's error handling path and messaging.
    """
    # Import root app only after the CLI tree has registered subcommands
    from ingenious.cli.main import app  # type: ignore

    # Patch the shim in the CLI module (not the underlying implementation)
    with patch(
        "ingenious.services.azure_search.cli.build_search_pipeline",
        side_effect=ValueError("semantic configuration name required"),
    ):
        result = CliRunner().invoke(app, _base_args(), env=_base_env())

    assert result.exit_code == 1
    # The CLI prints a Rich Panel with this prefix when ValueError bubbles up
    assert "Configuration failed:" in result.stdout
    assert "semantic configuration name required" in result.stdout


def test_cli_verbose_sets_component_loggers():
    """
    --verbose should set the Azure Search component loggers to DEBUG.
    We no-op the actual execution shim so the test stays fully offline.
    """
    from ingenious.cli.main import app  # type: ignore

    # Bring the CLI module in to access its __name__ for the logger list
    from ingenious.services.azure_search import cli as az_cli  # type: ignore

    # These are the logger names the CLI configures in setup_logging(verbose)
    logger_names = [
        "ingenious.services.azure_search.pipeline",
        "ingenious.services.azure_search.components.retrieval",
        "ingenious.services.azure_search.components.fusion",
        "ingenious.services.azure_search.components.generation",
        az_cli.__name__,  # CLI module logger itself
    ]

    # Start from a clean, non-DEBUG state to prove the effect
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    # Avoid running the real pipeline; we only want logging setup from CLI
    with patch(
        "ingenious.services.azure_search.cli._run_search_pipeline", return_value=None
    ):
        result = CliRunner().invoke(app, _base_args(verbose=True), env=_base_env())

    assert result.exit_code == 0

    # All component loggers should now be DEBUG
    for name in logger_names:
        assert logging.getLogger(name).level == logging.DEBUG, f"{name} not DEBUG"

    # Root logger should also be DEBUG when --verbose is used
    assert logging.getLogger().level == logging.DEBUG
