# ingenious/services/azure_search/tests/azure_search/test_provider_and_cli_flag.py

from typer.testing import CliRunner

from ingenious.services.azure_search import cli as CLI_MOD
from ingenious.services.azure_search.config import SearchConfig
from ingenious.services.azure_search.provider import AzureSearchProvider


class DummySettings:
    """Placeholder for IngeniousSettings; the builder stub ignores it."""

    pass


def _stub_pipeline_object():
    class _P:
        async def get_answer(self, q: str):
            # Minimal shape expected by the CLI rendering
            return {"answer": "", "source_chunks": []}

        async def close(self):
            return None

    return _P()


def test_provider_constructor_override_true(
    monkeypatch, config_no_semantic: SearchConfig
):
    captured = {}

    # Builder returns our fixture config
    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_config_from_settings",
        lambda _settings: config_no_semantic,
    )

    # Capture the cfg that reaches the pipeline factory
    def _capture_config(cfg):
        captured["cfg"] = cfg
        return _stub_pipeline_object()

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        _capture_config,
    )

    # Stub the rerank client used by the provider
    class _Dummy:
        async def close(self):  # pragma: no cover - trivial
            return None

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda _cfg: _Dummy(),
    )

    p = AzureSearchProvider(DummySettings(), enable_answer_generation=True)
    assert hasattr(p, "_pipeline")
    assert "cfg" in captured
    assert captured["cfg"].enable_answer_generation is True


def test_provider_constructor_override_none_preserves(
    monkeypatch, config_no_semantic: SearchConfig
):
    captured = {}

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_config_from_settings",
        lambda _settings: config_no_semantic,
    )

    def _capture_config(cfg):
        captured["cfg"] = cfg
        return _stub_pipeline_object()

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        _capture_config,
    )

    class _Dummy:
        async def close(self):  # pragma: no cover - trivial
            return None

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.make_search_client",
        lambda _cfg: _Dummy(),
    )

    _ = AzureSearchProvider(DummySettings(), enable_answer_generation=None)
    assert captured["cfg"].enable_answer_generation is False


def test_cli_generate_flag_and_env(monkeypatch):
    """
    Verify the CLI plumbs --generate and AZURE_SEARCH_ENABLE_GENERATION
    into SearchConfig.enable_answer_generation.

    Workaround for Click/Typer parsing quirk: place the positional QUERY
    immediately after the subcommand ('run') and BEFORE any options.
    Also patch both the shim and its lazy loader so our stub is always used.
    """
    received: list[SearchConfig] = []

    # The pipeline stub we want the CLI to use
    class _StubPipeline:
        async def get_answer(self, q: str):
            return {"answer": "", "source_chunks": []}

        async def close(self):
            pass

    def _shim(cfg: SearchConfig):
        received.append(cfg)
        return _StubPipeline()

    # Patch BOTH the exported shim and the lazy loader used inside the CLI
    monkeypatch.setattr(CLI_MOD, "build_search_pipeline", _shim)
    monkeypatch.setattr(CLI_MOD, "_get_build_pipeline_impl", lambda: _shim)

    runner = CliRunner()

    # Place QUERY ("q") right after 'run' so parsing is unambiguous
    base_args = [
        "run",
        "q",
        "--search-endpoint",
        "https://example.search.windows.net",
        "--search-key",
        "sk",
        "--search-index-name",
        "idx",
        "--openai-endpoint",
        "https://example.openai.azure.com",
        "--openai-key",
        "ok",
        "--embedding-deployment",
        "embed",
        "--generation-deployment",
        "gpt",
        "--no-semantic-ranking",
    ]

    # 1) Default (no --generate) => False
    res1 = runner.invoke(CLI_MOD.app, base_args)
    assert res1.exit_code == 0, res1.output
    assert received[-1].enable_answer_generation is False

    # 2) With flag => True
    res2 = runner.invoke(CLI_MOD.app, base_args + ["--generate"])
    assert res2.exit_code == 0, res2.output
    assert received[-1].enable_answer_generation is True

    # 3) With env var => True (no flag)
    env = {"AZURE_SEARCH_ENABLE_GENERATION": "true"}
    res3 = runner.invoke(CLI_MOD.app, base_args, env=env)
    assert res3.exit_code == 0, res3.output
    assert received[-1].enable_answer_generation is True
