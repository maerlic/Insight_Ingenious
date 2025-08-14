# tests/cli/test_azure_search_cli_features.py
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ingenious.services.azure_search.cli import app

runner = CliRunner()

BASE_ARGS = [
    "run",
    "--search-endpoint",
    "https://cli.search.windows.net",
    "--search-key",
    "cli-search-key",
    "--openai-endpoint",
    "https://cli.openai.azure.com",
    "--openai-key",
    "cli-openai-key",
    "--embedding-deployment",
    "cli-embed",
    "--generation-deployment",
    "cli-gen",
]


def test_azure_search_cli_load_custom_dat_prompt_file_success(tmp_path):
    prompt_content = "Custom DAT prompt content."
    prompt_file = tmp_path / "custom_prompt.txt"
    prompt_file.write_text(prompt_content)

    # NOTE: flags BEFORE the query; query LAST and after `--`
    args = BASE_ARGS + ["--dat-prompt-file", str(prompt_file), "--", "test query"]

    mock_run_pipeline = MagicMock()
    with patch(
        "ingenious.services.azure_search.cli._run_search_pipeline", mock_run_pipeline
    ):
        result = runner.invoke(app, args)

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    config_arg = mock_run_pipeline.call_args[0][0]
    assert config_arg.dat_prompt == prompt_content


def test_azure_search_cli_load_custom_dat_prompt_file_not_found():
    # NOTE: flags BEFORE the query; query LAST and after `--`
    args = BASE_ARGS + [
        "--dat-prompt-file",
        "/non/existent/file.txt",
        "--",
        "test query",
    ]

    mock_run_pipeline = MagicMock()
    with patch(
        "ingenious.services.azure_search.cli._run_search_pipeline", mock_run_pipeline
    ):
        result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "Error: DAT prompt file not found" in result.stdout
    mock_run_pipeline.assert_not_called()
