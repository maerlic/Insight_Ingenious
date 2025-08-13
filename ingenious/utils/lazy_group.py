from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple, TypeAlias

import click  # <- add this
import typer
from click import Command, Context
from typer.core import TyperGroup

LoadSpec: TypeAlias = Tuple[str, str, str]
LoadRegistry: TypeAlias = Dict[str, LoadSpec]


class LazyGroup(TyperGroup):
    """A Typer command group that lazy-loads its sub-CLIs."""

    _loaders: LoadRegistry = {
        "document-processing": (
            "ingenious.document_processing.cli",
            "doc_app",
            "document-processing",
        ),
        "dataprep": ("ingenious.dataprep.cli", "dataprep", "dataprep"),
        "chunk": ("ingenious.chunk.cli", "cli", "chunk"),
    }

    def list_commands(self, ctx: Context) -> List[str]:
        main_commands = super().list_commands(ctx)
        return sorted(set(main_commands + list(self._loaders.keys())))

    def _missing_extra_placeholder(self, name: str, extra: str) -> Command:
        # This is returned for help/completion. Executing it prints the install hint and exits 1.
        @click.command(
            name=name,
            help=f"[{extra}] extra not installed. "
            f"Install with: pip install 'insight-ingenious[{extra}]'",
        )
        def _cmd():
            typer.echo(
                f"\n[{extra}] extra not installed.\n"
                "Install with:\n\n"
                f"    pip install 'insight-ingenious[{extra}]'\n",
                err=True,
            )
            raise typer.Exit(1)

        return _cmd

    def get_command(self, ctx: Context, name: str) -> Optional[Command]:
        # First, any normal (already-registered) command/group
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd

        # Lazy entries
        if name not in self._loaders:
            return None

        module_path, attr_name, extra = self._loaders[name]
        try:
            module = importlib.import_module(module_path)
            sub_app = getattr(module, attr_name)
        except (ModuleNotFoundError, ImportError):
            # Do NOT raise here – return a placeholder so help can render.
            return self._missing_extra_placeholder(name, extra)

        # Convert Typer app to a Click command only once
        if isinstance(sub_app, Command):
            return sub_app
        return typer.main.get_command(sub_app)


__all__ = [
    "LazyGroup",
]
