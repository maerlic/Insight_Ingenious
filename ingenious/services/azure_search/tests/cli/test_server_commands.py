"""Test the server command-line interface commands.

This module contains unit tests for the `serve` command defined in
`ingenious.cli.server_commands`. It verifies that command-line arguments,
environment variables, and configuration file settings are correctly handled
and prioritized when launching the web server. The tests use mocks to isolate
the CLI logic from the actual server and configuration loading.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner, Result

import ingenious.cli.server_commands as server_module

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


runner: CliRunner = CliRunner()


def make_app_and_register() -> typer.Typer:
    """Create a Typer app and register the server commands onto it.

    This helper function centralizes the setup of the Typer application
    for tests, ensuring a consistent and clean app instance for each test case.
    """
    app: typer.Typer = typer.Typer()
    console: MagicMock = MagicMock()
    server_module.register_commands(app, console)
    return app


def stub_config(ip: str = "0.0.0.0", port: int = 80) -> SimpleNamespace:
    """Create a mock configuration object for testing.

    This function produces a simplified configuration object that mimics the
    structure accessed by the server startup logic, avoiding the need for a
    full configuration file load.
    """
    # Mimic the fields accessed by the server code
    web_conf = SimpleNamespace(ip_address=ip, port=port)
    cfg = SimpleNamespace(web_configuration=web_conf)
    return cfg


def test_serve_env_port_precedence(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test that the WEB_PORT environment variable sets the server port.

    This test verifies that when the `serve` command is run without a `--port`
    argument, the port is sourced from the `WEB_PORT` environment variable,
    demonstrating correct precedence of environment variables over defaults.
    """
    # Ensure a clean slate for env flags that the command may tweak
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # Set ENV before registering commands (default evaluated at declaration time)
    monkeypatch.setenv("WEB_PORT", "1234")

    app: typer.Typer = make_app_and_register()

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
        result: Result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0

        # config loaded and app constructed via seam
        get_cfg.assert_called_once()
        make_app_mock.assert_called_once_with(get_cfg.return_value)

        # uvicorn called with env-provided port (1234) and default host "0.0.0.0"
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        args, kwargs = uv_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 1234

        # Profiles.yml should be unset by default path
        assert "INGENIOUS_PROFILE_PATH" not in os.environ
        # LOADENV flipped
        assert os.environ.get("LOADENV") == "False"


def test_serve_cli_port_overrides_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test that CLI arguments for port and host override environment variables.

    This test confirms that when `--port` and `--host` arguments are provided
    to the `serve` command, their values take precedence over any conflicting
    settings from environment variables, ensuring direct user input is respected.
    """
    # Ensure a clean slate for env flags that the command may tweak
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # ENV present, but CLI overrides
    monkeypatch.setenv("WEB_PORT", "1234")

    app: typer.Typer = make_app_and_register()

    with (
        patch(
            "ingenious.cli.server_commands.get_config", return_value=stub_config()
        ) as get_cfg,
        patch(
            "ingenious.cli.server_commands.make_app", return_value=MagicMock()
        ) as make_app_mock,
        patch("ingenious.cli.server_commands.uvicorn.run") as uv_run,
    ):
        result: Result = runner.invoke(
            app, ["serve", "--port", "9999", "--host", "127.0.0.1"]
        )
        assert result.exit_code == 0

        # app constructed via seam
        make_app_mock.assert_called_once_with(get_cfg.return_value)

        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        args, kwargs = uv_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9999

        # Profiles path remains unset unless explicitly provided
        assert "INGENIOUS_PROFILE_PATH" not in os.environ


def test_serve_explicit_profile_path_handling(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Test that the --profile argument sets the profile path environment variable.

    This verifies that providing a path via the `--profile` command-line
    argument correctly sets the `INGENIOUS_PROFILE_PATH` environment
    variable, which is used by the configuration system to locate the
    `profiles.yml` file.
    """
    # Ensure a clean slate
    monkeypatch.delenv("INGENIOUS_PROFILE_PATH", raising=False)

    # Prepare a fake profiles.yml file
    profiles: Path = tmp_path / "profiles.yml"
    profiles.write_text("name: test")

    app: typer.Typer = make_app_and_register()

    with (
        patch(
            "ingenious.cli.server_commands.get_config", return_value=stub_config()
        ) as get_cfg,
        patch(
            "ingenious.cli.server_commands.make_app", return_value=MagicMock()
        ) as make_app_mock,
        patch("ingenious.cli.server_commands.uvicorn.run") as uv_run,
    ):
        result: Result = runner.invoke(app, ["serve", "--profile", str(profiles)])
        assert result.exit_code == 0

        # The command sets the env var when provided and exists
        assert os.environ.get("INGENIOUS_PROFILE_PATH") == str(profiles).replace(
            "\\", "/"
        )

        # ensure server called
        uv_run.assert_called_once()

        # app constructed via seam
        make_app_mock.assert_called_once_with(get_cfg.return_value)
