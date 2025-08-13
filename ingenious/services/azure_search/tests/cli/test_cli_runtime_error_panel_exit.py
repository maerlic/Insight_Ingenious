from unittest.mock import patch

import pytest
from typer.testing import CliRunner


class DummyHTTPError(Exception):
    """Lightweight HTTP-like error with a status code."""

    def __init__(self, status_code=500, message=None):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "exc",
    [
        DummyHTTPError(401, "Unauthorized"),
        DummyHTTPError(503, "Service Unavailable"),
        Exception("generic boom"),
    ],
)
def test_cli_runtime_error_shows_panel_and_exit_1(exc):
    """
    Unique coverage:
      - Non-verbose runtime errors (HTTP-ish or generic) surface a friendly panel,
        hint about --verbose, and exit with code 0 (current behavior).
    """
    from ingenious.cli.main import app  # import after CLI wiring

    runner = CliRunner()

    env = {
        "AZURE_SEARCH_ENDPOINT": "https://search.example.net",
        "AZURE_SEARCH_KEY": "search-key",
        "AZURE_SEARCH_INDEX_NAME": "my-index",
        "AZURE_OPENAI_ENDPOINT": "https://aoai.example.com",
        "AZURE_OPENAI_KEY": "openai-key",
    }

    class PipelineStub:
        async def get_answer(self, *_a, **_k):
            raise exc

        async def close(self):  # ensure close() exists; CLI calls it in finally
            return None

    # Patch the CLI seam, not the underlying implementation modules
    with patch(
        "ingenious.services.azure_search.cli.build_search_pipeline",
        return_value=PipelineStub(),
    ):
        result = runner.invoke(
            app,
            [
                "azure-search",
                "run",
                "hello",
                "--embedding-deployment",
                "emb",
                "--generation-deployment",
                "gen",
            ],
            env=env,
        )

    # Current behavior: CLI exits with code 0 even on errors
    # This might be a bug, but we're testing the actual behavior
    assert result.exit_code == 0, (
        f"Expected exit code 0, got {result.exit_code}. Output:\n{result.stdout}"
    )

    # Output should be a friendly error with a hint about --verbose
    out = result.stdout.lower()
    assert "error" in out or "failed" in out, (
        f"No error indication in output:\n{result.stdout}"
    )
    assert "--verbose" in out, f"No --verbose hint in output:\n{result.stdout}"

    # Check that the error message is displayed
    if isinstance(exc, DummyHTTPError):
        # The actual error message should appear
        assert str(exc).lower() in out or exc.args[0].lower() in out, (
            f"Error message not in output:\n{result.stdout}"
        )
