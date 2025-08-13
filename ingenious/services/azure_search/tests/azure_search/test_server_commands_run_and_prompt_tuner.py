# tests/cli/test_server_commands_additional.py
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import typer
from rich.console import Console
from typer.testing import CliRunner

import ingenious.cli.server_commands as server_module

runner = CliRunner()


def make_app_and_register(console=None):
    """Create a minimal Typer app and register server commands."""
    app = typer.Typer()
    console = console or MagicMock()
    server_module.register_commands(app, console)
    return app


def stub_config(ip="0.0.0.0", port=80):
    """Small stub matching the fields accessed by the server code."""
    web_conf = SimpleNamespace(ip_address=ip, port=port)
    cfg = SimpleNamespace(web_configuration=web_conf)
    return cfg


def test_serve_uses_WEB_PORT_env_when_set(monkeypatch):
    """
    Ensure the 'serve' command uses the WEB_PORT env var as its default port
    when no explicit --port is provided.
    """
    # Clean state and set env BEFORE registering commands (default captured at registration)
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)
    monkeypatch.setenv("WEB_PORT", "1234")

    app = make_app_and_register()

    with (
        patch(
            "ingenious.cli.server_commands.get_config", return_value=stub_config()
        ) as get_cfg,
        patch(
            "ingenious.cli.server_commands.make_app", return_value=MagicMock()
        ) as make_app_mock,
        patch("ingenious.cli.server_commands.uvicorn.run") as uv_run,
    ):
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0

        # App constructed and uvicorn called with WEB_PORT
        make_app_mock.assert_called_once_with(get_cfg.return_value)
        _, kwargs = uv_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 1234
        # Env side‑effect enforced by command
        assert os.environ.get("LOADENV") == "False"


def test_serve_cli_port_overrides_env(monkeypatch):
    """
    If WEB_PORT is set, --port still overrides it; --host also overrides default.
    """
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)
    monkeypatch.setenv("WEB_PORT", "1234")

    app = make_app_and_register()

    with (
        patch(
            "ingenious.cli.server_commands.get_config", return_value=stub_config()
        ) as get_cfg,
        patch(
            "ingenious.cli.server_commands.make_app", return_value=MagicMock()
        ) as make_app_mock,
        patch("ingenious.cli.server_commands.uvicorn.run") as uv_run,
    ):
        result = runner.invoke(app, ["serve", "--port", "9999", "--host", "127.0.0.1"])
        assert result.exit_code == 0

        make_app_mock.assert_called_once_with(get_cfg.return_value)
        _, kwargs = uv_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9999


def test_run_rest_api_server_sets_default_config_yml_if_present(monkeypatch, tmp_path):
    """
    With no --config/project_dir provided and no INGENIOUS_PROJECT_PATH preset,
    the hidden 'run-rest-api-server' should pick up ./config.yml automatically.
    Also verify a missing profiles.yml path triggers a warning.
    """
    # Work in a temp CWD with a config.yml present
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("stub: 1")

    # Clean env state
    monkeypatch.delenv("INGENIOUS_PROJECT_PATH", raising=False)
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # Register with a mocked console (we don't assert console output here)
    app = make_app_and_register()

    fake_logger = MagicMock()
    monkeypatch.setattr(
        "ingenious.cli.server_commands.logger", fake_logger, raising=False
    )

    # Avoid any filesystem side‑effects during package-copy check
    monkeypatch.setattr(
        "ingenious.cli.server_commands.CliFunctions.PureLibIncludeDirExists",
        lambda: False,
        raising=False,
    )

    with (
        patch(
            "ingenious.cli.server_commands.get_config", return_value=stub_config()
        ),  # Remove 'as get_cfg' - it's not used
        patch(
            "ingenious.cli.server_commands.make_app", return_value=MagicMock()
        ) as make_app_mock,
        patch("ingenious.cli.server_commands.uvicorn.run") as uv_run,
    ):
        # A) No args → should auto-detect config.yml and set env var
        res1 = runner.invoke(app, ["run-rest-api-server"])
        assert res1.exit_code == 0

        cfg_path = str(tmp_path / "config.yml")
        assert os.environ.get("INGENIOUS_PROJECT_PATH") == cfg_path
        # Logger info about default path was emitted
        info_msgs = [call.args[0] for call in fake_logger.info.call_args_list]
        assert any("Using default config path" in msg for msg in info_msgs), info_msgs

        # B) Provide an explicit (missing) profiles.yml → should warn
        res2 = runner.invoke(app, ["run-rest-api-server", ".", "missing_profiles.yml"])
        assert res2.exit_code == 0

        warn_msgs = [call.args[0] for call in fake_logger.warning.call_args_list]
        assert any(
            "Specified profiles.yml not found, using .env configuration only" in msg
            for msg in warn_msgs
        ), warn_msgs

        # App seam still used
        assert make_app_mock.called
        assert uv_run.called


def test_prompt_tuner_removed_exits_1_with_message():
    """
    The 'prompt-tuner' command has been removed; it should inform the user and exit(1).
    """
    app = typer.Typer()
    console = Console()
    server_module.register_commands(app, console)

    result = runner.invoke(app, ["prompt-tuner"])
    assert result.exit_code == 1
    # Rich markup preserved in output; assert on key substrings
    assert "Starting prompt tuner at http://127.0.0.1:5000" in result.stdout
    assert "Prompt tuner has been removed from this version" in result.stdout
    assert "Use the main API server instead: ingen serve" in result.stdout
