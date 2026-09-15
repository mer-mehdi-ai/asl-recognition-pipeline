"""Typed, file-backed configuration for every experiment.

A run is fully described by one YAML file: same file in, same pipeline out.
Nothing in the codebase reads a magic constant or an environment variable that
is not declared here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelName = Literal["landmark_rf", "cnn"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Base):
    """Where the images live and how many of them to use."""

    root: Path = Path("data/raw/asl_alphabet_train/asl_alphabet_train")
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_per_class: int | None = Field(
        default=None,
        description="Cap the number of images per class. Useful for smoke runs; None uses all.",
    )

    @field_validator("extensions")
    @classmethod
    def _lowercase(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(ext.lower() for ext in value)


class SplitConfig(_Base):
    """Train/test split policy.

    ``random``  - stratified random split. Optimistic on this dataset: consecutive
                  frames of the same recording end up on both sides of the split.
    ``blocked`` - per class, the last ``test_size`` fraction of the frame sequence is
                  held out. Approximates a session holdout and is the default here.
    See docs/evaluation.md.
    """

    strategy: Literal["random", "blocked"] = "blocked"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42


class LandmarkConfig(_Base):
    """MediaPipe Hands feature extraction."""

    static_image_mode: bool = True
    max_num_hands: int = 1
    min_detection_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    normalization: Literal["translate", "bbox"] = "bbox"
    features_path: Path = Path("data/interim/landmarks.parquet")


class RandomForestConfig(_Base):
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_jobs: int = -1
    seed: int = 42


class CnnConfig(_Base):
    image_size: int = 100
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 1e-3
    num_workers: int = 2
    device: Literal["auto", "cpu", "cuda"] = "auto"
    augment: bool = True
    seed: int = 42
    log_every: int = 30


class ExperimentConfig(_Base):
    """The whole experiment, as written on disk."""

    name: str
    model: ModelName
    runs_dir: Path = Path("runs")
    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    landmarks: LandmarkConfig = LandmarkConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    cnn: CnnConfig = CnnConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    def fingerprint(self) -> str:
        """Stable short hash of the configuration, stored in every run manifest."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


__all__ = [
    "CnnConfig",
    "DataConfig",
    "ExperimentConfig",
    "LandmarkConfig",
    "ModelName",
    "RandomForestConfig",
    "SplitConfig",
]
