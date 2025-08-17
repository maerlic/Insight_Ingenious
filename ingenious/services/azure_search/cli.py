"""Provides a CLI for the Advanced Azure AI Search service.

This module uses Typer to create a command-line interface for a multi-stage
search pipeline. It allows users to execute complex search queries against an
Azure AI Search index, leveraging features like retrieval, result fusion (DAT),
semantic reranking, and generative answering (RAG).

The main entry point is the `run` command, which accepts various configuration
options via command-line flags or environment variables. To simplify usage,
the CLI defaults to the `run` command, allowing users to pass a query
directly (e.g., `azure-search "my query"`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional

import click
import typer
from pydantic import SecretStr, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typer.core import TyperGroup

# ✅ Safe at import-time
from .config import DEFAULT_DAT_PROMPT, SearchConfig

# ── Lazy loader for the heavy pipeline ────────────────────────────────────────
_build_pipeline_impl: Optional[Callable[..., Any]] = None


def _get_build_pipeline_impl() -> Callable[..., Any]:
    """Lazily import and return the pipeline factory function.

    This avoids the high import cost of the full pipeline stack
    (e.g., torch, transformers) when the user is just asking for --help.
    """
    global _build_pipeline_impl
    if _build_pipeline_impl is None:
        from .components.pipeline import build_search_pipeline as _impl

        _build_pipeline_impl = _impl
    return _build_pipeline_impl


# Test‑patchable shim (the tests patch CLI_MOD.build_search_pipeline)
def build_search_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Provide a test-patchable shim for the pipeline factory.

    This function acts as an indirection layer, allowing tests to easily
    mock the pipeline construction process without interfering with the
    lazy-loading mechanism.
    """
    return _get_build_pipeline_impl()(*args, **kwargs)


# ── TyperGroup that safely forwards to the default "run" command ─────────────
class DefaultToRunTyperGroup(TyperGroup):
    """A TyperGroup that forwards to the 'run' command by default.

    This custom group enhances the user experience by allowing a query
    to be passed directly to the application, which then invokes the 'run'
    subcommand implicitly. It preserves standard behavior for explicit
    subcommands like 'run' or global options like '--help'.
    """

    def resolve_command(
        self, ctx: click.Context, args: List[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve the command, defaulting to 'run' if no subcommand is found.

        The logic first checks for explicit subcommands or group-level options.
        If none are present, it retrieves the 'run' command and passes the
        arguments to it. This makes the 'run' command the default action.
        """
        # If the first token is a group option (e.g., --help), let the group handle it.
        if args and args[0].startswith("-"):
            return super().resolve_command(ctx, args)

        # If an explicit subcommand is present, use it.
        if args:
            maybe_cmd = self.get_command(ctx, args[0])
            if maybe_cmd is not None:
                return args[0], maybe_cmd, args[1:]

        # Otherwise, forward to 'run' (with whatever args remain).
        cmd = self.get_command(ctx, "run")
        if cmd is not None:
            # If no args at all, Click will show "Missing argument 'QUERY'" with exit code 2.
            return "run", cmd, args

        # Fallback — should not occur since we define 'run'
        return super().resolve_command(ctx, args)


# Initialize Typer app (backed by our custom Typer group) and Rich console
app = typer.Typer(
    name="azure-search",
    help="CLI interface for the Ingenious Advanced Azure AI Search service.",
    cls=DefaultToRunTyperGroup,  # 👈 subclass of TyperGroup (satisfies tests)
    context_settings={
        "max_content_width": 200,
        # Allow routing `azure-search q --opts` → `run q --opts` safely
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
console = Console()

# Configure basic logging
logging.basicConfig(level=logging.WARNING)


# Helper to set logging levels for the service components
def setup_logging(verbose: bool) -> None:
    """Configure logging levels for the application components.

    This function sets the log level to DEBUG if verbose mode is enabled,
    or INFO otherwise, for key modules within the search service. This
    allows for controlled output during execution.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # List of loggers used in the service (adjust based on actual logger names if needed)
    loggers = [
        "ingenious.services.azure_search.pipeline",
        "ingenious.services.azure_search.components.retrieval",
        "ingenious.services.azure_search.components.fusion",
        "ingenious.services.azure_search.components.generation",
        __name__,  # Include CLI logger itself
    ]

    for logger_name in loggers:
        try:
            logging.getLogger(logger_name).setLevel(level)
        except Exception:
            # Handle cases where the package structure might differ
            pass

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def _run_search_pipeline(config: SearchConfig, query: str, verbose: bool) -> None:
    """Set up the asyncio event loop and execute the search pipeline.

    This helper encapsulates the asynchronous execution logic. It creates and
    runs an asyncio event loop to manage the pipeline's async operations,
    ensuring proper resource initialization and cleanup.
    """

    async def _async_run() -> None:
        """Build, run, and clean up the search pipeline asynchronously."""
        pipeline: Any = None
        try:
            # Build the pipeline using the factory
            pipeline = build_search_pipeline(config)

            # Make a status line visible in captured output for tests
            console.print("Executing Advanced Search Pipeline", markup=False)

            # Execute the pipeline
            result: dict[str, Any]
            with console.status(
                "[bold green]Executing Advanced Search Pipeline (L1 -> DAT -> L2 -> RAG)...",
                spinner="dots",
            ):
                result = await pipeline.get_answer(query)

            # Display Results
            answer: str = result.get("answer", "No answer generated.")
            sources: list[dict[str, Any]] = result.get("source_chunks", [])

            console.print(
                Panel(
                    Markdown(answer),
                    title="[bold green]:robot: Answer[/bold green]",
                    border_style="green",
                )
            )

            # Display Sources
            console.print(f"\n[bold]Sources Used ({len(sources)}):[/bold]")
            for i, source in enumerate(sources):
                score: float | str = source.get("_final_score", "N/A")
                content_sample: str = source.get(config.content_field, "")[:250] + "..."

                score_display = (
                    f"{score:.4f}" if isinstance(score, float) else str(score)
                )

                console.print(
                    Panel(
                        content_sample,
                        title=f"[bold cyan]Chunk {i + 1} "
                        f"(Score: {score_display} | Type: {source.get('_retrieval_type', 'N/A')})[/bold cyan]",
                        border_style="cyan",
                        expand=False,
                    )
                )

        except ValueError as ve:
            console.print(
                Panel(
                    f"Configuration failed: {ve}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
            # Tests expect exit(1) on configuration errors
            raise typer.Exit(code=1)
        except Exception as e:
            # Handle runtime errors
            if verbose:
                console.print_exception(show_locals=True)
            console.print(
                Panel(
                    f"Pipeline execution failed: {e}\n[dim]Run with --verbose for details.[/dim]",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
        finally:
            # Ensure clients are closed
            if pipeline:
                await pipeline.close()
                logging.info("Pipeline clients closed.")

    asyncio.run(_async_run())


# Keep a minimal callback so `azure-search --help` prints group help
@app.callback()
def _callback() -> None:
    """Provide the main entry point for the Typer application group.

    This callback ensures that the group-level help message is displayed
    correctly. The logic for defaulting to the 'run' command is handled
    by the custom DefaultToRunTyperGroup class.
    """
    # Default-to-run is implemented in DefaultToRunTyperGroup.resolve_command
    return None


# ── Subcommand: explicit `run` (options kept identical, with aliases) ─────────
@app.command(
    name="run",
    # Accept options after positional args for tests like: run "q" --search-endpoint ...
    context_settings={"allow_interspersed_args": True},
)
def run_search(
    # ── Azure AI Search Configuration ─────────────────────────────────────────
    search_endpoint: str = typer.Option(
        ...,
        "--search-endpoint",
        "-se",
        envvar="AZURE_SEARCH_ENDPOINT",
        help="Azure AI Search Endpoint URL.",
    ),
    search_key: str = typer.Option(
        ...,
        "--search-key",
        "-sk",
        envvar="AZURE_SEARCH_KEY",
        help="Azure AI Search API Key.",
        prompt=True,
        hide_input=True,
    ),
    search_index_name: str | None = typer.Option(
        None,
        "--search-index-name",
        "-si",
        help="Azure AI Search index name to use.",
        envvar="AZURE_SEARCH_INDEX_NAME",
        show_envvar=True,
    ),
    # ── Azure OpenAI Configuration ────────────────────────────────────────────
    openai_endpoint: str = typer.Option(
        ...,
        "--openai-endpoint",
        "-oe",
        envvar="AZURE_OPENAI_ENDPOINT",
        help="Azure OpenAI Endpoint URL.",
    ),
    openai_key: str = typer.Option(
        ...,
        "--openai-key",
        "-ok",
        envvar="AZURE_OPENAI_KEY",
        help="Azure OpenAI API Key.",
        prompt=True,
        hide_input=True,
    ),
    embedding_deployment: str = typer.Option(
        ...,
        "--embedding-deployment",
        "-ed",
        envvar="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        help="Embedding model deployment name.",
    ),
    generation_deployment: str = typer.Option(
        ...,
        "--generation-deployment",
        "-gd",
        envvar="AZURE_OPENAI_GENERATION_DEPLOYMENT",
        help="Generation model deployment name (used for DAT and RAG).",
    ),
    # ── Pipeline Behavior Configuration ───────────────────────────────────────
    top_k_retrieval: int = typer.Option(
        20,
        "--top-k-retrieval",
        "-k",
        help="Number of initial results to fetch (K).",
    ),
    use_semantic_ranking: bool = typer.Option(
        True,
        "--semantic-ranking/--no-semantic-ranking",
        help="Enable/Disable Azure Semantic Ranking (L2).",
    ),
    semantic_config_name: str | None = typer.Option(
        None,
        "--semantic-config",
        "-sc",
        envvar="AZURE_SEARCH_SEMANTIC_CONFIG",
        help="Semantic configuration name (required if using semantic ranking).",
    ),
    top_n_final: int = typer.Option(
        5,
        "--top-n-final",
        "-n",
        help="Number of final chunks for generation (N).",
    ),
    openai_version: str = typer.Option(
        "2024-02-01",
        "--openai-version",
        "-ov",
        help="Azure OpenAI API Version.",
    ),
    dat_prompt_file: str | None = typer.Option(
        None,
        "--dat-prompt-file",
        "-dp",
        help="Path to a custom DAT prompt file (overrides default).",
    ),
    generate: bool = typer.Option(
        False,
        "--generate/--no-generate",
        envvar="AZURE_SEARCH_ENABLE_GENERATION",
        help="Enable/disable final answer generation (default: disabled).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
    # ── Positional query (must be last when invoking) ─────────────────────────
    query: str = typer.Argument(..., help="The search query string."),
) -> None:
    """Execute the advanced AI search pipeline.

    This command orchestrates the full search process: retrieval, Dynamic
    Alternating Transformation (DAT) for query fusion, semantic reranking,
    and final answer generation. It gathers all necessary configuration
    from command-line options or environment variables, validates them, and
    initiates the pipeline execution.
    """
    setup_logging(verbose)

    console.print(f"\nStarting search for: '[bold]{query}[/bold]'\n", markup=False)

    # Guardrail for tests: if semantic ranking is enabled, name must be supplied
    if use_semantic_ranking and not semantic_config_name:
        typer.echo(
            "Error: Semantic ranking is enabled but no semantic configuration name was provided.\n"
            "Supply --semantic-config or set AZURE_SEARCH_SEMANTIC_CONFIG."
        )
        raise typer.Exit(code=1)

    # Handle DAT prompt loading
    dat_prompt: str = DEFAULT_DAT_PROMPT
    if dat_prompt_file:
        try:
            with open(dat_prompt_file, "r") as f:
                dat_prompt = f.read()
            logging.info(f"Loaded custom DAT prompt from {dat_prompt_file}")
        except FileNotFoundError:
            # Plain, stable message for tests to assert reliably
            typer.echo("Error: DAT prompt file not found")
            raise typer.Exit(code=1)

    # Build the configuration object
    try:
        config = SearchConfig(
            search_endpoint=search_endpoint,
            search_key=SecretStr(search_key),
            search_index_name=search_index_name,
            semantic_configuration_name=semantic_config_name,
            openai_endpoint=openai_endpoint,
            openai_key=SecretStr(openai_key),
            openai_version=openai_version,
            embedding_deployment_name=embedding_deployment,
            generation_deployment_name=generation_deployment,
            top_k_retrieval=top_k_retrieval,
            use_semantic_ranking=use_semantic_ranking,
            top_n_final=top_n_final,
            dat_prompt=dat_prompt,
            enable_answer_generation=generate,
        )
    except ValidationError as e:
        console.print(f"[bold red]Configuration Validation Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    _run_search_pipeline(config, query, verbose)
