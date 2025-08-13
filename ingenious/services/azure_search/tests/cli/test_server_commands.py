# tests/cli/test_server_commands.py
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

import ingenious.cli.server_commands as server_module

runner = CliRunner()


def make_app_and_register():
    app = typer.Typer()
    console = MagicMock()
    server_module.register_commands(app, console)
    return app


def stub_config(ip="0.0.0.0", port=80):
    # Mimic the fields accessed by the server code
    web_conf = SimpleNamespace(ip_address=ip, port=port)
    cfg = SimpleNamespace(web_configuration=web_conf)
    return cfg


def test_serve_env_port_precedence(tmp_path, monkeypatch):
    # Ensure a clean slate for env flags that the command may tweak
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # Set ENV before registering commands (default evaluated at declaration time)
    monkeypatch.setenv("WEB_PORT", "1234")

    app = make_app_and_register()

    # Patch get_config, make_app seam, uvicorn.run
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

        # config loaded and app constructed via seam
        get_cfg.assert_called_once()
        make_app_mock.assert_called_once_with(get_cfg.return_value)

        # uvicorn called with env-provided port (1234) and default host "0.0.0.0"
        args, kwargs = uv_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 1234

        # Profiles.yml should be unset by default path
        assert "INGENIOUS_PROFILE_PATH" not in os.environ
        # LOADENV flipped
        assert os.environ.get("LOADENV") == "False"


def test_serve_cli_port_overrides_env(tmp_path, monkeypatch):
    # Ensure a clean slate for env flags that the command may tweak
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # ENV present, but CLI overrides
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

        # app constructed via seam
        make_app_mock.assert_called_once_with(get_cfg.return_value)

        args, kwargs = uv_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9999

        # Profiles path remains unset unless explicitly provided
        assert "INGENIOUS_PROFILE_PATH" not in os.environ


def test_serve_explicit_profile_path_handling(tmp_path, monkeypatch):
    # Ensure a clean slate
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # Prepare a fake profiles.yml file
    profiles = tmp_path / "profiles.yml"
    profiles.write_text("name: test")

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
        result = runner.invoke(app, ["serve", "--profile", str(profiles)])
        assert result.exit_code == 0

        # The command sets the env var when provided and exists
        assert os.environ.get("INGENIOUS_PROFILE_PATH") == str(profiles).replace(
            "\\", "/"
        )

        # ensure server called
        uv_run.assert_called_once()

        # app constructed via seam
        make_app_mock.assert_called_once_with(get_cfg.return_value)
