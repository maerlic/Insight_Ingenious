# ingenious/services/azure_search/tests/cli/test_azure_search_help_contract.py

from typer.testing import CliRunner


def test_azure_search_run_help_lists_required_flags():
    """
    Help contract: the azure-search run subcommand must list all required flags
    for wiring Azure Search + Azure OpenAI.
    """
    from ingenious.cli.main import app  # import after app wiring

    runner = CliRunner()
    res = runner.invoke(app, ["azure-search", "run", "--help"])
    assert res.exit_code == 0, res.stdout

    out = res.stdout
    required = [
        "--search-endpoint",
        "--search-key",
        "--search-index-name",
        "--openai-endpoint",
        "--openai-key",
        "--embedding-deployment",
        "--generation-deployment",
    ]
    missing = [flag for flag in required if flag not in out]
    assert not missing, f"Missing flags in help: {missing}\n\n{out}"
