"""Immutable run directories.

Every training or evaluation command writes into a fresh directory named
``<UTC timestamp>-<experiment name>-<config fingerprint>``. A run directory is
never reused: re-running an experiment produces a new one, so a metric on disk
can always be traced back to the exact config, code revision and environment
that produced it.

    runs/20260915T101500123Z-cnn-baseline-4f1c9a2b8d10/
      manifest.json          who / what / when / which commit
      config.yaml            the exact config used
      metrics.json           everything measured, merged across steps
      artifacts/             model weights, encoders, figures, reports
      logs/run.log           stdout of the command
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asl import __version__
from asl.config import ExperimentConfig

RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{9}Z-[A-Za-z0-9._-]+-[0-9a-f]{12}$")


class RunExistsError(RuntimeError):
    """Raised when a run directory already exists: runs are never overwritten."""


class RunNotFoundError(FileNotFoundError):
    pass


def timestamp() -> str:
    """UTC stamp with millisecond precision.

    Seconds are not enough: two runs of the same config launched in the same
    second would collide, and a run directory is never reused.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"


def git_revision(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform dependent
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


@dataclass(frozen=True)
class Run:
    """Handle on one run directory."""

    path: Path

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def artifacts(self) -> Path:
        return self.path / "artifacts"

    @property
    def logs(self) -> Path:
        return self.path / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.json"

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def config(self) -> ExperimentConfig:
        return ExperimentConfig.from_yaml(self.config_path)

    def metrics(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            return {}
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))

    def log_metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Merge ``payload`` into metrics.json. Existing keys are kept unless overwritten."""
        current = self.metrics()
        current.update(payload)
        self.metrics_path.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")
        return current

    def artifact(self, *parts: str) -> Path:
        target = self.artifacts.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


def create_run(config: ExperimentConfig, runs_dir: Path | None = None) -> Run:
    """Create a fresh run directory. Never reuses an existing one."""
    root = Path(runs_dir or config.runs_dir)
    stamp = timestamp()
    run_id = f"{stamp}-{config.name}-{config.fingerprint()}"
    path = root / run_id
    if path.exists():
        raise RunExistsError(f"run directory already exists: {path}")
    (path / "artifacts").mkdir(parents=True)
    (path / "logs").mkdir(parents=True)

    run = Run(path)
    config.to_yaml(run.config_path)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config.name,
        "model": config.model,
        "config_fingerprint": config.fingerprint(),
        "git_revision": git_revision(Path(__file__).resolve().parents[2]),
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    run.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run


def list_runs(runs_dir: Path) -> list[Run]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return [
        Run(p)
        for p in sorted(runs_dir.iterdir())
        if p.is_dir() and RUN_ID_PATTERN.match(p.name) and (p / "manifest.json").exists()
    ]


def resolve_run(run_ref: str, runs_dir: Path) -> Run:
    """Resolve ``latest``, a run id, or a path to a run directory."""
    runs_dir = Path(runs_dir)
    if run_ref == "latest":
        runs = list_runs(runs_dir)
        if not runs:
            raise RunNotFoundError(f"no run found under {runs_dir}")
        return runs[-1]

    candidate = Path(run_ref)
    if candidate.is_dir() and (candidate / "manifest.json").exists():
        return Run(candidate)

    candidate = runs_dir / run_ref
    if candidate.is_dir() and (candidate / "manifest.json").exists():
        return Run(candidate)

    raise RunNotFoundError(f"unknown run: {run_ref}")


__all__ = [
    "Run",
    "RunExistsError",
    "RunNotFoundError",
    "create_run",
    "git_revision",
    "list_runs",
    "resolve_run",
    "timestamp",
]
