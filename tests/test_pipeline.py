"""The pipeline test: from a folder of images to a run directory with metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from asl.pipeline import build_catalog, train, train_landmark_pipeline


def _reader(path: str):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


@pytest.fixture(autouse=True)
def _patch_reader(monkeypatch):
    """Read images with Pillow so the pipeline test does not need OpenCV."""
    monkeypatch.setattr("asl.data.landmarks._default_reader", _reader)


def test_build_catalog_applies_the_split(config):
    catalog = build_catalog(config)
    assert set(catalog.columns) >= {"path", "label", "order", "split"}
    assert set(catalog["split"]) == {"train", "test"}


def test_landmark_pipeline_produces_a_complete_run(config, fake_extractor):
    run = train_landmark_pipeline(
        config, extractor=fake_extractor, reuse_features=False, log=lambda message: None
    )

    metrics = run.metrics()
    assert metrics["dataset"]["n_classes"] == 4
    assert metrics["split"]["n_train"] > 0
    assert metrics["hand_detection_rate"]["overall"] == 1.0
    assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
    assert metrics["split_strategy"] == "blocked"

    assert (run.artifacts / "model" / "landmark_rf.joblib").exists()
    assert (run.artifacts / "eval" / "classification_report.csv").exists()
    assert (run.artifacts / "catalog.csv").exists()
    assert json.loads(run.manifest_path.read_text())["model"] == "landmark_rf"
    assert config.landmarks.features_path.exists()


def test_pipeline_can_reuse_cached_features(config, fake_extractor):
    train_landmark_pipeline(
        config, extractor=fake_extractor, reuse_features=False, log=lambda message: None
    )
    first_calls = fake_extractor.calls

    train_landmark_pipeline(
        config, extractor=fake_extractor, reuse_features=True, log=lambda message: None
    )
    assert fake_extractor.calls == first_calls, "cached features should not be recomputed"


def test_trained_run_can_be_loaded_back_and_predict(config, fake_extractor):
    from asl.registry import load_recognizer_from_ref

    run = train_landmark_pipeline(
        config, extractor=fake_extractor, reuse_features=False, log=lambda message: None
    )
    recognizer, resolved = load_recognizer_from_ref(run.run_id, config.runs_dir)
    assert resolved.run_id == run.run_id
    assert len(recognizer.classes) == 4

    recognizer._extractor = fake_extractor
    prediction = recognizer.predict_image(np.full((32, 32, 3), 40, dtype=np.uint8))
    assert prediction.label in recognizer.classes


def test_train_dispatches_on_the_config(config, fake_extractor):
    run = train(config, log=lambda message: None, extractor=fake_extractor, reuse_features=False)
    assert run.manifest()["model"] == "landmark_rf"


def test_train_rejects_an_unknown_backend(config):
    broken = config.model_copy(update={"model": "svm"})
    with pytest.raises(ValueError, match="unknown model"):
        train(broken, log=lambda message: None)
