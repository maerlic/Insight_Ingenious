# ingenious/cli/search_commands.py
from __future__ import annotations

import typer
from rich.console import Console


def register_commands(app: typer.Typer, console: Console) -> None:
    # Import the sub-app lazily but directly
    from ingenious.services.azure_search import cli as search_cli

    # Mount it once, under the required name
    app.add_typer(search_cli.app, name="azure-search")
