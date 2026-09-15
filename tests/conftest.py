from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from asl.config import ExperimentConfig
from asl.data.landmarks import FEATURE_COLUMNS, FEATURE_DIM

CLASSES = ["A", "B", "del", "space"]


@pytest.fixture
def image_root(tmp_path: Path) -> Path:
    """A miniature ImageFolder tree: 4 classes x 10 deterministic images."""
    root = tmp_path / "dataset"
    rng = np.random.default_rng(0)
    for class_index, label in enumerate(CLASSES):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for i in range(10):
            array = np.full((32, 32, 3), 20 * class_index, dtype=np.uint8)
            array += rng.integers(0, 12, size=array.shape, dtype=np.uint8)
            Image.fromarray(array).save(class_dir / f"{label}{i + 1}.jpg")
    return root


@pytest.fixture
def config(tmp_path: Path, image_root: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "unit-test",
            "model": "landmark_rf",
            "runs_dir": str(tmp_path / "runs"),
            "data": {"root": str(image_root), "extensions": [".jpg"]},
            "split": {"strategy": "blocked", "test_size": 0.3, "seed": 0},
            "landmarks": {"features_path": str(tmp_path / "landmarks.parquet")},
            "random_forest": {"n_estimators": 25, "seed": 0},
        }
    )


class FakeExtractor:
    """Deterministic stand-in for MediaPipe: features encode the class.

    Keeps the pipeline tests fast and independent of the native MediaPipe build,
    while still exercising the real feature table and the real classifier.
    """

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray | None:
        self.calls += 1
        base = float(np.mean(image_bgr)) / 255.0
        if self.fail_on is not None and self.fail_on in str(base):  # pragma: no cover
            return None
        rng = np.random.default_rng(int(base * 1000))
        return (base + rng.normal(0, 0.01, size=FEATURE_DIM)).astype(np.float32)


@pytest.fixture
def fake_extractor() -> FakeExtractor:
    return FakeExtractor()


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A ready-made, separable feature table for classifier-level tests."""
    rng = np.random.default_rng(1)
    rows = []
    for class_index, label in enumerate(CLASSES):
        for i in range(12):
            vector = rng.normal(class_index, 0.05, size=FEATURE_DIM).astype(np.float32)
            row = {
                "path": f"/tmp/{label}{i}.jpg",
                "label": label,
                "order": i,
                "split": "train" if i < 8 else "test",
            }
            row.update(dict(zip(FEATURE_COLUMNS, vector.tolist(), strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def jpeg_bytes() -> bytes:
    import io

    image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
