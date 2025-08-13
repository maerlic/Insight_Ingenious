# -- coding: utf-8 --

import typer
from rich.console import Console
from typer.testing import CliRunner

from ingenious.cli.workflow_commands import register_commands


def _make_app():
    # Use a colorless console so assertions don't have to deal with ANSI codes
    console = Console(no_color=True, force_terminal=False, color_system=None)
    app = typer.Typer()
    register_commands(app, console)
    return app


def test_workflows_all_lists_expected_non_deprecated_entries():
    app = _make_app()
    runner = CliRunner()

    # No argument -> "all" listing
    result = runner.invoke(app, ["workflows"])
    assert result.exit_code == 0
    out = result.stdout

    # Present (non-deprecated workflows)
    assert "classification-agent" in out
    assert "bike-insights" in out
    assert "knowledge-base-agent" in out
    assert "sql-manipulation-agent" in out

    # Absent (explicitly deprecated legacy names)
    assert "knowledge_base_agent" not in out
    assert "sql_manipulation_agent" not in out


def test_workflows_detail_shows_example_curl_and_note():
    app = _make_app()
    runner = CliRunner()

    result = runner.invoke(app, ["workflows", "bike-insights"])
    assert result.exit_code == 0
    out = result.stdout

    # Basic sections
    assert "📋 BIKE-INSIGHTS REQUIREMENTS" in out or "BIKE-INSIGHTS REQUIREMENTS" in out
    assert "Description:" in out
    assert "Category:" in out
    assert "External Services Needed:" in out
    assert "Configuration Required:" in out

    # Note section appears and includes the key guidance
    assert "Note:" in out or "⚠️  Note:" in out
    assert "recommended first workflow" in out.lower()

    # Example curl block is printed for bike-insights
    assert "🧪 TEST COMMAND:" in out or "TEST COMMAND:" in out
    assert "curl -X POST http://localhost:80/api/v1/chat" in out
    assert '"conversation_flow": "bike-insights"' in out


def test_workflows_unknown_prints_available_list():
    app = _make_app()
    runner = CliRunner()

    result = runner.invoke(app, ["workflows", "does-not-exist"])
    assert result.exit_code == 0
    out = result.stdout

    # Clear error and available list
    assert "Unknown workflow: does-not-exist" in out
    assert "Available workflows:" in out

    # Shows only non-deprecated entries in the "available" list
    assert "classification-agent" in out
    assert "bike-insights" in out
    assert "knowledge-base-agent" in out
    assert "sql-manipulation-agent" in out

    # Deprecated names are not listed
    assert "knowledge_base_agent" not in out
    assert "sql_manipulation_agent" not in out
