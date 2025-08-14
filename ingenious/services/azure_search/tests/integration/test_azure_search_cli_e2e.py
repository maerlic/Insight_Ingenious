# -*- coding: utf-8 -*-
import os

import pytest
from typer.testing import CliRunner

REQUIRED_ENV = [
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_GENERATION_DEPLOYMENT",
]


def _have_env():
    return all(os.getenv(k) for k in REQUIRED_ENV)


@pytest.mark.azure_integration
@pytest.mark.skipif(not _have_env(), reason="Azure integration env vars not set")
def test_cli_e2e_runs_pipeline_and_prints_status():
    from ingenious.cli.main import app

    # Remove mix_stderr parameter - not supported in newer versions
    runner = CliRunner()

    res = runner.invoke(app, ["azure-search", "run", "sanity question"])
    assert res.exit_code == 0, (res.stdout or "") + (res.stderr or "")
    # A couple of tolerant status markers; change to your actual phrasing if you prefer
    combined = (res.stdout or "") + (res.stderr or "")
    assert ("Executing Advanced Search Pipeline" in combined) or (
        "Starting search for:" in combined
    )
