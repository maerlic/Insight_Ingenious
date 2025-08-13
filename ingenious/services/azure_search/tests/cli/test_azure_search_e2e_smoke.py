# tests/cli/test_azure_search_e2e_smoke.py
from unittest.mock import patch

from typer.testing import CliRunner


def test_azure_search_run_smoke_success():
    from ingenious.cli.main import app

    runner = CliRunner()

    env = {
        "AZURE_SEARCH_ENDPOINT": "https://s",
        "AZURE_SEARCH_KEY": "k",
        "AZURE_SEARCH_INDEX_NAME": "i",
        "AZURE_OPENAI_ENDPOINT": "https://oai",
        "AZURE_OPENAI_KEY": "ok",
    }

    args = [
        "azure-search",
        "run",
        "q",
        "--embedding-deployment",
        "emb",
        "--generation-deployment",
        "gen",
    ]

    with patch("ingenious.services.azure_search.cli._run_search_pipeline") as mock_run:
        result = runner.invoke(app, args, env=env)

    assert result.exit_code == 0
    # The CLI prints a “Starting search for” line; we can assert it’s present
    assert "Starting search for: '[bold]q[/bold]'" in result.stdout
    mock_run.assert_called_once()
