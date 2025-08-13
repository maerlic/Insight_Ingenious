# tests/cli/test_search_commands_wrapper.py
import typer
from rich.console import Console
from typer.testing import CliRunner


def test_search_commands_register_adds_typer():
    from ingenious.cli.search_commands import register_commands

    app = typer.Typer()
    console = Console()
    register_commands(app, console)

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "azure-search" in result.stdout

    # Ensure the run command exists
    result = runner.invoke(app, ["azure-search", "run", "--help"])
    assert result.exit_code == 0
    assert "--search-endpoint" in result.stdout
