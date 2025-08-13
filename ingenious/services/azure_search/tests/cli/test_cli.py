# -- coding: utf-8 --

"""
FILE TEST PLAN

    setup_logging(): verbose vs non-verbose levels across service loggers + root

    _run_search_pipeline(): success path shows panels and closes pipeline; error paths:
        factory ValueError → printed config error panel
        runtime Exception → printed error, optional trace with --verbose

    Typer command 'run':
        env-var based defaults parsed into SearchConfig
        options override env vars, including toggles and custom DAT prompt file
        missing prompt file → exit(1) and do not run pipeline
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from ingenious.services.azure_search.cli import _run_search_pipeline, app, setup_logging
from ingenious.services.azure_search.config import DEFAULT_DAT_PROMPT, SearchConfig

runner = CliRunner()
CLI_MOD = "ingenious.services.azure_search.cli"


def test_setup_logging_verbose():
    with patch("logging.getLogger") as gl:
        lg = MagicMock()
        gl.return_value = lg
        setup_logging(verbose=True)
        # root called once at end w/o name
        assert lg.setLevel.called
        for call in lg.setLevel.call_args_list:
            assert call.args[0] == logging.DEBUG


def test_setup_logging_non_verbose():
    with patch("logging.getLogger") as gl:
        lg = MagicMock()
        gl.return_value = lg
        setup_logging(verbose=False)
        for call in lg.setLevel.call_args_list:
            assert call.args[0] == logging.INFO


def _patch_build_pipeline(mock_instance):
    return patch(f"{CLI_MOD}.build_search_pipeline", mock_instance)


def test_run_search_pipeline_success(config: SearchConfig, capsys):
    mock_pipe = MagicMock()
    mock_pipe.get_answer = AsyncMock(
        return_value={
            "answer": "A",
            "source_chunks": [
                {
                    "id": "S",
                    "content": "X" * 400,
                    "_final_score": 3.51234,
                    "_retrieval_type": "hyb",
                }
            ],
        }
    )
    mock_pipe.close = AsyncMock()
    with _patch_build_pipeline(MagicMock(return_value=mock_pipe)):
        _run_search_pipeline(config, "q", verbose=False)
    out = capsys.readouterr().out
    assert "Executing Advanced Search Pipeline" in out
    assert "Answer" in out or "A" in out
    assert "Sources Used (1)" in out
    assert "3.5123" in out  # rounded
    mock_pipe.close.assert_awaited()


def test_run_search_pipeline_config_error(config: SearchConfig, capsys):
    with _patch_build_pipeline(MagicMock(side_effect=ValueError("bad sem"))):
        # New contract: _run_search_pipeline must exit(1) on config errors
        with pytest.raises(typer.Exit) as ei:
            _run_search_pipeline(config, "q", verbose=False)
    assert ei.value.exit_code == 1
    out = capsys.readouterr().out
    assert "Configuration failed: bad sem" in out


def test_run_search_pipeline_runtime_error_verbose(config: SearchConfig):
    mock_pipe = MagicMock()
    mock_pipe.get_answer = AsyncMock(side_effect=RuntimeError("boom"))
    mock_pipe.close = AsyncMock()
    with (
        _patch_build_pipeline(MagicMock(return_value=mock_pipe)),
        patch(f"{CLI_MOD}.console.print_exception") as pe,
    ):
        _run_search_pipeline(config, "q", verbose=True)
        pe.assert_called_once()


ENV = {
    "AZURE_SEARCH_ENDPOINT": "https://s",
    "AZURE_SEARCH_KEY": "sk",
    "AZURE_SEARCH_INDEX_NAME": "idx",
    "AZURE_OPENAI_ENDPOINT": "https://o",
    "AZURE_OPENAI_KEY": "ok",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "emb",
    "AZURE_OPENAI_GENERATION_DEPLOYMENT": "gen",
    "AZURE_SEARCH_SEMANTIC_CONFIG": "sem",
}


def test_cli_run_env_parsing(tmp_path):
    with (
        patch(f"{CLI_MOD}._run_search_pipeline") as rp,
        patch(f"{CLI_MOD}.setup_logging") as sl,
    ):
        res = runner.invoke(app, ["what?"], env=ENV)
        assert res.exit_code == 0
        sl.assert_called_once_with(False)
        cfg, q, verbose = rp.call_args[0]
        assert isinstance(cfg, SearchConfig)
        assert q == "what?"
        assert verbose is False
        assert cfg.semantic_configuration_name == "sem"
        assert cfg.dat_prompt == DEFAULT_DAT_PROMPT


def test_cli_run_options_override(tmp_path):
    with (
        patch(f"{CLI_MOD}._run_search_pipeline") as rp,
        patch(f"{CLI_MOD}.setup_logging"),
    ):
        res = runner.invoke(
            app,
            [
                "q",
                "--top-k-retrieval",
                "50",
                "--top-n-final",
                "10",
                "--no-semantic-ranking",
                "--verbose",
                "--search-endpoint",
                "https://override",
            ],
            env=ENV,
        )
        assert res.exit_code == 0
        cfg = rp.call_args[0][0]
        assert cfg.top_k_retrieval == 50
        assert cfg.top_n_final == 10
        assert cfg.use_semantic_ranking is False
        assert cfg.search_endpoint == "https://override"


def test_cli_custom_dat_prompt_success(tmp_path):
    p = tmp_path / "dat.txt"
    p.write_text("CUSTOM")
    with patch(f"{CLI_MOD}._run_search_pipeline") as rp:
        res = runner.invoke(app, ["q", "--dat-prompt-file", str(p)], env=ENV)
        assert res.exit_code == 0
        assert rp.call_args[0][0].dat_prompt == "CUSTOM"


def test_cli_custom_dat_prompt_missing():
    with patch(f"{CLI_MOD}._run_search_pipeline") as rp:
        res = runner.invoke(app, ["q", "--dat-prompt-file", "missing.txt"], env=ENV)
        assert res.exit_code == 1
        assert "DAT prompt file not found" in res.stdout
        rp.assert_not_called()
