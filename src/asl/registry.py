"""Load the right backend out of a run directory.

The manifest says which backend was trained; the caller never has to know.
"""

from __future__ import annotations

from pathlib import Path

from asl.models.base import Recognizer
from asl.runs import Run, resolve_run

BACKENDS = ("landmark_rf", "cnn")


def load_recognizer(run: Run) -> Recognizer:
    backend = run.manifest()["model"]
    directory = run.artifacts / "model"
    if backend == "landmark_rf":
        from asl.models.landmark_rf import LandmarkRecognizer

        return LandmarkRecognizer.load(directory)
    if backend == "cnn":
        from asl.models.cnn import CnnRecognizer

        return CnnRecognizer.load(directory)
    raise ValueError(f"unknown backend in manifest: {backend}")


def load_recognizer_from_ref(run_ref: str, runs_dir: str | Path = "runs") -> tuple[Recognizer, Run]:
    run = resolve_run(run_ref, Path(runs_dir))
    return load_recognizer(run), run


__all__ = ["BACKENDS", "load_recognizer", "load_recognizer_from_ref"]
