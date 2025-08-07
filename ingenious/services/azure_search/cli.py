# Insight_Ingenious/ingenious/services/azure_search/cli.py

import typer
import asyncio
import logging
from typing import Optional
from pydantic import SecretStr, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Import the public facade and necessary configuration elements
try:
    from ingenious.services.azure_search import SearchConfig, build_search_pipeline
    from ingenious.services.azure_search.config import DEFAULT_DAT_PROMPT
except ImportError:
    # Fallback for local development/testing
    # Note: When running locally, ensure config.py and pipeline.py are accessible
    from config import SearchConfig, DEFAULT_DAT_PROMPT
    from pipeline import build_search_pipeline


# Initialize Typer app and Rich console
app = typer.Typer(
    name="search", help="CLI interface for the Ingenious Advanced Azure AI Search service."
)
console = Console()

# Configure basic logging
logging.basicConfig(level=logging.WARNING)

# Helper to set logging levels for the service components
def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    
    # List of loggers used in the service (adjust based on actual logger names if needed)
    loggers = [
        "ingenious.services.azure_search.pipeline",
        "ingenious.services.azure_search.components.retrieval",
        "ingenious.services.azure_search.components.fusion",
        "ingenious.services.azure_search.components.generation",
        __name__ # Include CLI logger itself
    ]
    
    for logger_name in loggers:
        try:
            logging.getLogger(logger_name).setLevel(level)
        except Exception:
            # Handle cases where the package structure might differ
            pass
    
    if verbose:
         logging.getLogger().setLevel(logging.DEBUG)


def _run_search_pipeline(config: SearchConfig, query: str, verbose: bool):
    """Helper function to manage the asyncio event loop and run the pipeline."""
    async def _async_run():
        pipeline = None
        try:
            # Build the pipeline using the factory
            pipeline = build_search_pipeline(config)
            
            # Execute the pipeline
            with console.status("[bold green]Executing Advanced Search Pipeline (L1 -> DAT -> L2 -> RAG)...", spinner="dots"):
                result = await pipeline.get_answer(query)
            
            # Display Results
            answer = result.get("answer", "No answer generated.")
            sources = result.get("source_chunks", [])

            console.print(Panel(
                Markdown(answer),
                title="[bold green]:robot: Answer[/bold green]",
                border_style="green"
            ))

            # Display Sources
            console.print(f"\n[bold]Sources Used ({len(sources)}):[/bold]")
            for i, source in enumerate(sources):
                score = source.get('_final_score', 'N/A')
                content_sample = source.get(config.content_field, '')[:250] + "..."
                
                score_display = f"{score:.4f}" if isinstance(score, float) else str(score)

                console.print(Panel(
                    content_sample,
                    title=f"[bold cyan]Chunk {i+1} (Score: {score_display} | Type: {source.get('_retrieval_type', 'N/A')})[/bold cyan]",
                    border_style="cyan",
                    expand=False
                ))

        except ValueError as ve:
            # Handle configuration validation errors from the factory
             console.print(Panel(
                f"Configuration failed: {ve}",
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
        except Exception as e:
            # Handle runtime errors
            if verbose:
                console.print_exception(show_locals=True)
            console.print(Panel(
                f"Pipeline execution failed: {e}\n[dim]Run with --verbose for details.[/dim]",
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
        finally:
            # Ensure clients are closed
            if pipeline:
                await pipeline.close()
                logging.info("Pipeline clients closed.")

    asyncio.run(_async_run())

@app.command(name="run")
def run_search(
    query: str = typer.Argument(..., help="The search query string."),
    
    # Azure AI Search Configuration
    search_endpoint: str = typer.Option(..., envvar="AZURE_SEARCH_ENDPOINT", help="Azure AI Search Endpoint URL."),
    search_key: str = typer.Option(..., envvar="AZURE_SEARCH_KEY", help="Azure AI Search API Key.", prompt=True, hide_input=True),
    search_index_name: str = typer.Option(..., envvar="AZURE_SEARCH_INDEX_NAME", help="Target index name."),
    
    # Azure OpenAI Configuration
    openai_endpoint: str = typer.Option(..., envvar="AZURE_OPENAI_ENDPOINT", help="Azure OpenAI Endpoint URL."),
    openai_key: str = typer.Option(..., envvar="AZURE_OPENAI_KEY", help="Azure OpenAI API Key.", prompt=True, hide_input=True),
    embedding_deployment: str = typer.Option(..., envvar="AZURE_OPENAI_EMBEDDING_DEPLOYMENT", help="Embedding model deployment name."),
    generation_deployment: str = typer.Option(..., envvar="AZURE_OPENAI_GENERATION_DEPLOYMENT", help="Generation model deployment name (used for DAT and RAG)."),
    
    # Pipeline Behavior Configuration
    top_k_retrieval: int = typer.Option(20, help="Number of initial results to fetch (K)."),
    use_semantic_ranking: bool = typer.Option(True, "--semantic-ranking/--no-semantic-ranking", help="Enable/Disable Azure Semantic Ranking (L2)."),
    semantic_config_name: Optional[str] = typer.Option(None, envvar="AZURE_SEARCH_SEMANTIC_CONFIG", help="Semantic configuration name (required if using semantic ranking)."),
    top_n_final: int = typer.Option(5, help="Number of final chunks for generation (N)."),
    openai_version: str = typer.Option("2024-02-01", help="Azure OpenAI API Version."),
    dat_prompt_file: Optional[str] = typer.Option(None, help="Path to a custom DAT prompt file (overrides default)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """
    Execute the advanced multi-stage AI search pipeline (Retrieve -> DAT Fuse -> Semantic Rerank -> Generate).
    """
    setup_logging(verbose)

    console.print(f"\n:mag: Starting search for: '[bold]{query}[/bold]'\n")

    # Handle DAT prompt loading
    dat_prompt = DEFAULT_DAT_PROMPT
    if dat_prompt_file:
        try:
            with open(dat_prompt_file, 'r') as f:
                dat_prompt = f.read()
            logging.info(f"Loaded custom DAT prompt from {dat_prompt_file}")
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] DAT prompt file not found: {dat_prompt_file}")
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
            dat_prompt=dat_prompt
        )
    except ValidationError as e:
        console.print(f"[bold red]Configuration Validation Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    _run_search_pipeline(config, query, verbose)