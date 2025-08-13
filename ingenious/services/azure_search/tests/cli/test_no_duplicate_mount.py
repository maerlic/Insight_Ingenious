# tests/cli/test_no_duplicate_mount.py
from typer.testing import CliRunner


def test_azure_search_appears_once_in_help():
    from ingenious.cli.main import app

    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    count = res.stdout.count("azure-search")
    assert count == 1, f"azure-search appears {count} times in help:\n{res.stdout}"
