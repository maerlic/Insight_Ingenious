# ingenious/services/azure_search/tests/cli/test_cli_invalid_semantic.py

import pytest
from typer.testing import CliRunner


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Current CLI prints a config error panel but exits 0 when "
        "use_semantic_ranking=True and no semantic config name is supplied. "
        "Desired behavior: exit code 1."
    ),
)
def test_cli_invalid_semantic_requires_name_exits_1():
    """
    Semantic config contradiction:
    With semantic ranking enabled (default) and no semantic configuration name provided,
    the CLI should treat it as a fatal config error and exit(1).
    """
    from ingenious.cli.main import app  # import after app wiring

    env = {
        # Azure Search (no semantic config provided)
        "AZURE_SEARCH_ENDPOINT": "https://search.example.net",
        "AZURE_SEARCH_KEY": "sk",
        "AZURE_SEARCH_INDEX_NAME": "idx",
        # Azure OpenAI
        "AZURE_OPENAI_ENDPOINT": "https://oai.example.com",
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
        # note: NOT passing --no-semantic-ranking and NOT setting AZURE_SEARCH_SEMANTIC_CONFIG
    ]

    res = CliRunner().invoke(app, args, env=env)
    # Desired contract:
    assert res.exit_code == 1, res.stdout
    assert "semantic" in res.stdout.lower()
