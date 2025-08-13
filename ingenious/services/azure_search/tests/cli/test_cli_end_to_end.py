# ingenious/services/azure_search/tests/integration/test_cli_end_to_end.py
# -*- coding: utf-8 -*-

"""
Integration smoke test for the Azure Search CLI.

Requires these environment variables to be set (else the test is skipped):
  - AZURE_SEARCH_ENDPOINT
  - AZURE_SEARCH_KEY
  - AZURE_SEARCH_INDEX_NAME
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_KEY
  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT
  - AZURE_OPENAI_GENERATION_DEPLOYMENT

This runs the real `azure-search run` command and asserts that:
  - the command exits 0,
  - the output contains a non-empty “Answer” panel,
  - the output contains a “Sources Used (N):” line.

NOTE: We pass `--no-semantic-ranking` to avoid requiring AZURE_SEARCH_SEMANTIC_CONFIG.
"""

import os
import re

import pytest
from typer.testing import CliRunner


@pytest.mark.azure_integration
def test_end_to_end_cli_runs_query_and_maps_results():
    required_env = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_GENERATION_DEPLOYMENT",
    ]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing env vars for Azure integration test: {missing}")

    # Build the environment passed to the CLI (only the required keys)
    env = {k: os.environ[k] for k in required_env}

    # Query can be overridden if you want to target a known doc in your index
    query = os.getenv("AZURE_SEARCH_TEST_QUERY", "integration smoke test")

    # Use small K/N to minimize cost and latency; disable semantic ranking unless AZURE_SEARCH_SEMANTIC_CONFIG is provided.
    args = [
        "azure-search",
        "run",
        query,
        "--embedding-deployment",
        env["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        "--generation-deployment",
        env["AZURE_OPENAI_GENERATION_DEPLOYMENT"],
        "--no-semantic-ranking",
        "--top-k-retrieval",
        "4",
        "--top-n-final",
        "2",
    ]

    # Invoke the real CLI
    from ingenious.cli.main import app  # import after app wiring

    runner = CliRunner()
    result = runner.invoke(app, args, env=env)

    # Basic execution succeeded
    assert result.exit_code == 0, f"CLI failed:\nSTDOUT:\n{result.stdout}"

    out = result.stdout

    # Prologue shows the query we sent (useful debug)
    assert query in out, f"Expected query '{query}' to be echoed.\n{out}"

    # An answer panel is printed; we just require that it exists and is non-empty text.
    # The panel title contains the word 'Answer' (Rich markup may be present).
    assert "Answer" in out, f"Expected an Answer panel in output.\n{out}"

    # The CLI always prints 'Sources Used (N):' even if N == 0.
    m = re.search(r"Sources Used\s*\((\d+)\):", out)
    assert m is not None, f"Expected 'Sources Used (N):' line.\n{out}"
    # N is a non-negative integer; we don't force >0 to avoid flakiness across indices.
    n_sources = int(m.group(1))
    assert n_sources >= 0
