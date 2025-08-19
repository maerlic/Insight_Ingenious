# Insight_Ingenious/ingenious/cli/main.py
"""
Main CLI application setup for Insight Ingenious.

This module contains the core CLI application configuration and imports
all command modules to register them with the typer app.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.theme import Theme

os.environ.setdefault("COLUMNS", "200")
from ingenious.utils.lazy_group import LazyGroup

# ────────────────────────────────────────────────────────────────────────────
# Typer application singleton
# ────────────────────────────────────────────────────────────────────────────

app: typer.Typer = typer.Typer(
    cls=LazyGroup,
    no_args_is_help=True,  # `ingen` → show help when no args
    pretty_exceptions_show_locals=False,  # cleaner tracebacks in production
    help="""
🚀 Insight Ingenious - GenAI Accelerator

A powerful framework for building and deploying AI agent workflows.

Quick Start:
  ingen init                    # Initialize a new project
  ingen serve                   # Start the API server
  ingen workflows               # List available workflows

Common Commands:
  init, serve, test, workflows, prompt-tuner

Data Processing:
  dataprep, document-processing

Get help for any command with: ingen <command> --help
    """.strip(),
)


# Ensure bare invocation / --help always exit 0 and show subcommands
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    if ctx.invoked_subcommand is None and (
        not ctx.args or any(h in ctx.args for h in ("-h", "--help"))
    ):
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "dark_orange",
        "danger": "bold red",
        "error": "bold red",
        "debug": "khaki1",
    }
)
console = Console(theme=custom_theme)

# Import the new command registry
from .registry import get_registry

# Initialize command registry with the console
registry = get_registry(console)

# Import command modules to register them with the app
from . import (
    help_commands,
    project_commands,
    search_commands,
    server_commands,
    test_commands,
    workflow_commands,
)

# Register commands from each module
server_commands.register_commands(app, console)
project_commands.register_commands(app, console)
test_commands.register_commands(app, console)
workflow_commands.register_commands(app, console)
help_commands.register_commands(app, console)
search_commands.register_commands(app, console)

# Discover additional commands
registry.discover_commands(
    [
        "ingenious.cli.help_commands",
        "ingenious.cli.project_commands",
        "ingenious.cli.server_commands",
        "ingenious.cli.test_commands",
        "ingenious.cli.workflow_commands",
        "ingenious.cli.search_commands",
    ]
)

# Register commands with typer app
registry.register_typer_commands(app)

if __name__ == "__main__":
    app()
