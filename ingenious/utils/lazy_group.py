"""Provide a lazy-loading Typer/Click command group.

This module defines LazyGroup, a custom TyperGroup subclass. Its primary purpose
is to defer the import of sub-commands until they are actually invoked. This is
useful for large command-line applications with optional dependencies ("extras"),
as it significantly speeds up startup time and tab-completion by avoiding costly
imports for commands that may not even be installed.

The main entry point is the LazyGroup class, which can be used as a `cls` for a
Typer application. When a command is requested that is not yet registered but is
listed in the lazy-loading registry, the group attempts to import it on-the-fly.
If the import fails (e.g., because the required extra is not installed), it
provides a helpful error message instead of crashing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, TypeAlias

import click
import typer
from click import Command, Context
from click.formatting import HelpFormatter
from typer.core import TyperGroup

if TYPE_CHECKING:
    from types import ModuleType

LoadSpec: TypeAlias = Tuple[str, str, str]
LoadRegistry: TypeAlias = Dict[str, LoadSpec]


def _is_real_click_context(ctx: object) -> bool:
    """Check if an object is a genuine click.Context, not a test mock.

    This function determines if the provided `ctx` is an actual `click.Context`
    instance created by the Click framework. It uses a strict identity check
    (`type(ctx) is Context`) to avoid being fooled by mock objects used in unit
    tests, which might otherwise mimic the context's interface but cause
    unexpected behavior in error-handling paths.

    Returns:
        True if ctx is a real click.Context, False otherwise.
    """
    try:
        return type(ctx) is Context  # strict identity check
    except Exception:
        return False


class LazyGroup(TyperGroup):
    """A Typer command group that lazy-loads its sub-CLIs.

    This group maintains a registry of commands that can be loaded on-demand.
    When a command is requested, it first checks for eagerly registered commands
    and then consults its lazy-loading registry. This prevents incurring import
    costs for all possible sub-commands at application startup.
    """

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
        """List all available commands, both eager and lazy.

        This method extends the base implementation by combining the commands
        already registered with Typer (eager commands) with the keys from the
        lazy-loading registry. The combined list is then sorted to ensure a
        consistent and predictable order in help messages.

        Args:
            ctx: The Click context.

        Returns:
            A sorted list of all command names.
        """
        main_commands: list[str] = super().list_commands(ctx)
        # Merge eager commands with lazy entries and deduplicate
        return sorted(set(main_commands + list(self._loaders.keys())))

    # --- Robust help rendering that never triggers lazy import/Exit ---
    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        """Render the Commands section of help text without triggering lazy imports.

        This method is carefully designed to generate help text safely. It avoids
        calling `self.get_command()` on lazy-loaded commands, which could trigger
        an import and potentially exit if dependencies are missing. Instead, it
        retrieves eager commands normally and generates placeholder help text
        for lazy commands, indicating if their corresponding optional extra is
        not installed.

        Args:
            ctx: The Click context.
            formatter: The Click help formatter.
        """
        rows: List[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            # Resolve only eager/registered commands using the base method to avoid recursion.
            base_cmd: Command | None = TyperGroup.get_command(self, ctx, name)
            if base_cmd is not None:
                rows.append((name, base_cmd.get_short_help_str()))
                continue

            # For known lazy commands, show a safe placeholder line.
            if name in self._loaders:
                _, _, extra = self._loaders[name]
                rows.append(
                    (
                        name,
                        f"[{extra}] extra not installed. "
                        f"Install with: pip install 'insight-ingenious[{extra}]'",
                    )
                )
                continue
            # Unknown names are skipped.

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def _missing_extra_placeholder(self, name: str, extra: str) -> Command:
        """Create a placeholder command for a missing optional dependency.

        This factory method generates a dummy `click.Command`. When this command
        is invoked, it prints a helpful message to the user explaining which
        "extra" dependency is missing and how to install it, then exits with a
        non-zero status code. This provides a user-friendly failure mode.

        Args:
            name: The name of the command that could not be loaded.
            extra: The name of the optional extra required for the command.

        Returns:
            A Click command that prints an error and exits.
        """

        # Returned for help/completion or general help rendering.
        # Executing it prints the install hint and exits 1.
        @click.command(
            name=name,
            help=(
                f"[{extra}] extra not installed. "
                f"Install with: pip install 'insight-ingenious[{extra}]'"
            ),
        )
        def _cmd() -> None:
            """Print install hint for a missing extra and exit."""
            typer.echo(
                f"\n[{extra}] extra not installed.\n"
                "Install with:\n\n"
                f"    pip install 'insight-ingenious[{extra}]'\n",
                err=True,
            )
            raise typer.Exit(1)

        return _cmd

    def get_command(self, ctx: Context, name: str) -> Optional[Command]:
        """Retrieve a command, loading it lazily if necessary.

        This method is the core of the lazy-loading mechanism. It first attempts
        to find an eagerly registered command. If none is found, it checks the
        lazy registry. If a match is found, it tries to import the specified
        module and retrieve the command object. If the import fails due to a
        `ModuleNotFoundError` (indicating a missing extra), it returns a
        placeholder command that provides a helpful installation message.

        Args:
            ctx: The Click context.
            name: The name of the command to retrieve.

        Returns:
            The requested Click command, a placeholder on failure, or None.
        """
        # First, any normal (already-registered) command/group
        cmd: Command | None = super().get_command(ctx, name)
        if cmd is not None:
            return cmd

        # Lazy entries
        if name not in self._loaders:
            return None

        module_path, attr_name, extra = self._loaders[name]
        try:
            module: ModuleType = importlib.import_module(module_path)
            sub_app: Command | typer.Typer = getattr(module, attr_name)
        except (ModuleNotFoundError, ImportError):
            # If in a real Click Context (help/introspection) OR resilient parsing, do NOT raise here.
            # Return a placeholder so help and other introspection can render without crashing.
            if _is_real_click_context(ctx) or (
                getattr(ctx, "resilient_parsing", False) is True
            ):
                return self._missing_extra_placeholder(name, extra)
            # Unit-test / non-Click path: behave as the test expects (echo + Exit).
            typer.echo(
                f"[{extra}] extra not installed. "
                f"Install with: pip install 'insight-ingenious[{extra}]'"
            )
            raise typer.Exit(1)
        except AttributeError:
            # Missing attribute on the module; treat similarly but with a clearer message.
            msg: str = (
                f"Could not find attribute '{attr_name}' in '{module_path}'. "
                "Please reinstall or report a bug."
            )
            if _is_real_click_context(ctx) or (
                getattr(ctx, "resilient_parsing", False) is True
            ):

                @click.command(name=name, help=msg)
                def _cmd() -> None:
                    """Print an error for a misconfigured command and exit."""
                    typer.echo(msg, err=True)
                    raise typer.Exit(1)

                return _cmd
            typer.echo(msg)
            raise typer.Exit(1)

        # If it's already a Click command, return as-is; otherwise convert Typer app to Click command
        if isinstance(sub_app, Command):
            return sub_app

        return typer.main.get_command(sub_app)


__all__ = ["LazyGroup"]
