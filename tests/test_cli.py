"""Tests for the CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_help_shows_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "train" in result.output
    assert "serve" in result.output
    assert "status" in result.output


def test_status_runs():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Service" in result.output
    assert "Training Backend" in result.output


def test_init_creates_env_file(tmp_path):
    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0
    env_file = tmp_path / ".env"
    assert env_file.exists()
    content = env_file.read_text()
    assert "LLM_MODEL=" in content
    assert "NATS_URL=" in content


def test_init_skips_existing_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING=true\n")
    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "already exists" in result.output
    assert env_file.read_text() == "EXISTING=true\n"


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "standalone" in result.output
    assert "openrlhf" in result.output
