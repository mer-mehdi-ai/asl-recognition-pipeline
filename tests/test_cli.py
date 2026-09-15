from __future__ import annotations

from typer.testing import CliRunner

from asl import __version__
from asl.cli import app
from asl.runs import create_run

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_every_command_group():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("data", "catalog", "train", "runs", "predict", "explain", "serve"):
        assert command in result.stdout


def test_catalog_command_prints_the_split(tmp_path, config):
    config_path = config.to_yaml(tmp_path / "config.yaml")
    result = runner.invoke(app, ["catalog", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "blocked" in result.stdout


def test_runs_list_on_an_empty_directory(tmp_path):
    result = runner.invoke(app, ["runs", "list", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no run" in result.stdout


def test_runs_list_and_show(config):
    run = create_run(config)
    run.log_metrics({"test": {"accuracy": 0.5, "macro_f1": 0.4}, "split_strategy": "blocked"})

    listed = runner.invoke(app, ["runs", "list", "--runs-dir", str(config.runs_dir)])
    assert listed.exit_code == 0
    assert "landmark_rf" in listed.stdout

    shown = runner.invoke(
        app, ["runs", "show", "--run", run.run_id, "--runs-dir", str(config.runs_dir)]
    )
    assert shown.exit_code == 0
    assert "config_fingerprint" in shown.stdout
