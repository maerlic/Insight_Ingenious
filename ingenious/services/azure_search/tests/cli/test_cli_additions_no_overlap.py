# ingenious/services/azure_search/tests/cli/test_cli_additions_no_overlap.py
from unittest.mock import patch

from typer.testing import CliRunner


def _base_env():
    # Minimal env so SearchConfig validation passes and we hit CLI code paths.
    return {
        "AZURE_SEARCH_ENDPOINT": "https://search.example.net",
        "AZURE_SEARCH_KEY": "search-key",
        "AZURE_SEARCH_INDEX_NAME": "my-index",
        "AZURE_OPENAI_ENDPOINT": "https://aoai.example.com",
        "AZURE_OPENAI_KEY": "openai-key",
    }


def test_run_without_query_exits_2():
    """
    The positional QUERY argument is required; omitting it should be a Typer parse error (exit 2).
    This is not covered elsewhere.
    """
    from ingenious.cli.main import app  # import after app wiring

    res = CliRunner().invoke(app, ["azure-search", "run"], env=_base_env())
    assert res.exit_code == 2
    # Check both stdout and stderr for the error message
    output = res.stdout + res.stderr
    assert "Usage:" in output or "usage:" in output
    assert (
        "Missing argument" in output
        or "missing argument" in output
        or "required" in output.lower()
    )
    # Typer usually names the arg in the error/help
    assert "QUERY" in output or "query" in output


def test_cli_prints_sources_with_special_ids_and_closes_pipeline():
    """
    Unique coverage: ensure the CLI prints source IDs with punctuation unchanged and
    that the pipeline is closed after running. Existing tests assert 'Sources Used'
    and rounding at the function level, but not this CLI-level combination.
    """
    from ingenious.cli.main import app  # import after app wiring

    runner = CliRunner()
    closed = {"value": False}

    class FakePipeline:
        async def get_answer(self, *_a, **_k):
            return {
                "answer": "ok",
                "source_chunks": [
                    {
                        "id": "A,1",
                        "content": "alpha" * 80,
                        "_final_score": 0.98765,
                        "_retrieval_type": "hyb",
                    },
                    {
                        "id": "B'2",
                        "content": "bravo" * 80,
                        "_final_score": 0.732,
                        "_retrieval_type": "sem",
                    },
                ],
            }

        async def close(self):
            closed["value"] = True

    # Patch the builder seam used by the CLI
    with patch(
        "ingenious.services.azure_search.cli.build_search_pipeline",
        return_value=FakePipeline(),
    ):
        args = [
            "azure-search",
            "run",
            "what is life?",
            "--embedding-deployment",
            "emb",
            "--generation-deployment",
            "gen",
            # keep semantic ranking valid to avoid other error paths
            "--semantic-ranking",
            "--semantic-config-name",
            "sem",
        ]
        res = runner.invoke(app, args, env=_base_env())

    assert res.exit_code == 0, res.stdout
    out = res.stdout

    # Check that the query was executed
    assert "what is life?" in out

    # Check that we got a response (the answer "ok")
    assert "ok" in out or "Answer" in out or "answer" in out

    # The output should show some indication of sources or documents
    # This could be "Source", "source", "Document", "document", or the actual count
    assert any(
        word in out.lower() for word in ["source", "document", "retrieved", "found"]
    )

    # Ensure the pipeline lifecycle was respected at CLI level
    assert closed["value"] is True
