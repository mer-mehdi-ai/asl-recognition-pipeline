from __future__ import annotations

import pytest

from asl.runs import RunExistsError, RunNotFoundError, create_run, list_runs, resolve_run


def test_create_run_writes_manifest_and_config(config):
    run = create_run(config)
    manifest = run.manifest()
    assert manifest["experiment"] == config.name
    assert manifest["config_fingerprint"] == config.fingerprint()
    assert manifest["model"] == config.model
    assert run.config() == config
    assert run.artifacts.is_dir()
    assert run.logs.is_dir()


def test_run_id_encodes_experiment_and_fingerprint(config):
    run = create_run(config)
    assert config.name in run.run_id
    assert run.run_id.endswith(config.fingerprint())


def test_runs_are_never_overwritten(config, monkeypatch):
    run = create_run(config)
    frozen = run.run_id.split("-")[0]
    monkeypatch.setattr("asl.runs.timestamp", lambda: frozen)
    with pytest.raises(RunExistsError):
        create_run(config)


def test_timestamp_has_millisecond_precision():
    from asl.runs import timestamp

    stamp = timestamp()
    assert len(stamp) == len("20260915T101500123Z")
    assert stamp.endswith("Z")


def test_log_metrics_merges_instead_of_replacing(config):
    run = create_run(config)
    run.log_metrics({"a": 1})
    run.log_metrics({"b": 2})
    assert run.metrics() == {"a": 1, "b": 2}


def test_resolve_latest_returns_the_newest_run(config):
    first = create_run(config)
    second = create_run(config.model_copy(update={"name": "second"}))
    resolved = resolve_run("latest", config.runs_dir)
    assert resolved.run_id in {first.run_id, second.run_id}
    assert resolved.run_id == sorted([first.run_id, second.run_id])[-1]


def test_resolve_by_id_and_by_path(config):
    run = create_run(config)
    assert resolve_run(run.run_id, config.runs_dir).path == run.path
    assert resolve_run(str(run.path), config.runs_dir).path == run.path


def test_resolve_unknown_run_raises(config):
    create_run(config)
    with pytest.raises(RunNotFoundError):
        resolve_run("does-not-exist", config.runs_dir)


def test_list_runs_ignores_foreign_directories(config):
    create_run(config)
    (config.runs_dir / "scratch").mkdir()
    assert len(list_runs(config.runs_dir)) == 1


def test_list_runs_on_missing_directory(tmp_path):
    assert list_runs(tmp_path / "absent") == []
