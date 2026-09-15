from __future__ import annotations

import numpy as np
import pytest

from asl.config import RandomForestConfig
from asl.data.landmarks import FEATURE_COLUMNS, FEATURE_DIM
from asl.evaluate import compute_metrics, confusion_frame, top_confusions
from asl.models.landmark_rf import LandmarkRecognizer, train_random_forest


def test_random_forest_learns_a_separable_problem(feature_frame):
    model, classes = train_random_forest(feature_frame, RandomForestConfig(n_estimators=40, seed=0))
    test = feature_frame[feature_frame["split"] == "test"]
    predictions = model.predict(test[FEATURE_COLUMNS].to_numpy())
    assert (predictions == test["label"].to_numpy()).mean() > 0.9
    assert classes == sorted(feature_frame["label"].unique())


def test_recognizer_aligns_probabilities_with_the_class_list(feature_frame):
    model, classes = train_random_forest(feature_frame, RandomForestConfig(n_estimators=40, seed=0))
    recognizer = LandmarkRecognizer(model=model, classes=classes)
    vector = feature_frame.loc[feature_frame["label"] == "B", FEATURE_COLUMNS].iloc[0].to_numpy()
    prediction = recognizer.predict_features(vector)
    assert prediction.label == "B"
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.scores).issubset(set(classes))
    assert prediction.as_dict()["label"] == "B"


def test_recognizer_roundtrips_through_disk(feature_frame, tmp_path):
    model, classes = train_random_forest(feature_frame, RandomForestConfig(n_estimators=40, seed=0))
    recognizer = LandmarkRecognizer(model=model, classes=classes)
    recognizer.save(tmp_path / "model")
    restored = LandmarkRecognizer.load(tmp_path / "model")

    vector = feature_frame.loc[:, FEATURE_COLUMNS].iloc[0].to_numpy()
    assert restored.classes == recognizer.classes
    assert restored.predict_features(vector).label == recognizer.predict_features(vector).label


def test_recognizer_returns_none_when_no_hand_is_detected(feature_frame):
    model, classes = train_random_forest(feature_frame, RandomForestConfig(n_estimators=10, seed=0))
    recognizer = LandmarkRecognizer(model=model, classes=classes, extractor=lambda image: None)
    assert recognizer.predict_image(np.zeros((8, 8, 3), dtype=np.uint8)) is None


def test_training_without_train_rows_fails_loudly(feature_frame):
    only_test = feature_frame.assign(split="test")
    with pytest.raises(ValueError, match="no training rows"):
        train_random_forest(only_test, RandomForestConfig())


def test_metrics_report_accuracy_macro_f1_and_worst_classes():
    classes = ["A", "B", "C"]
    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "A", "B", "C", "C", "B"]
    metrics = compute_metrics(y_true, y_pred, classes)
    assert metrics["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert metrics["n_classes"] == 3
    assert metrics["weakest_classes"][0]["label"] in {"B", "C"}


def test_top_confusions_points_at_the_real_mistake():
    classes = ["M", "N", "S"]
    y_true = ["M"] * 5 + ["N"] * 5
    y_pred = ["N"] * 5 + ["N"] * 5
    matrix = confusion_frame(y_true, y_pred, classes)
    worst = top_confusions(matrix)[0]
    assert (worst["true"], worst["predicted"], worst["count"]) == ("M", "N", 5)


def test_evaluation_artifacts_are_written(tmp_path):
    from asl.evaluate import evaluation_artifacts

    classes = ["A", "B"]
    metrics = evaluation_artifacts(["A", "B", "A"], ["A", "B", "B"], classes, tmp_path)
    assert (tmp_path / "classification_report.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert metrics["n_samples"] == 3


@pytest.mark.parametrize("image_size", [64, 100])
def test_cnn_forward_shape_and_roundtrip(tmp_path, image_size):
    torch = pytest.importorskip("torch")
    from asl.config import CnnConfig
    from asl.models.cnn import CnnRecognizer, build_model

    classes = ["A", "B", "space"]
    config = CnnConfig(image_size=image_size, device="cpu")
    model = build_model(len(classes), image_size)
    logits = model(torch.zeros(2, 3, image_size, image_size))
    assert logits.shape == (2, len(classes))

    recognizer = CnnRecognizer(model=model, classes=classes, config=config)
    recognizer.save(tmp_path / "cnn")
    restored = CnnRecognizer.load(tmp_path / "cnn")
    assert restored.classes == classes

    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    before = recognizer.predict_image(frame)
    after = restored.predict_image(frame)
    assert before.label == after.label
    assert before.confidence == pytest.approx(after.confidence, abs=1e-5)


def test_cnn_training_step_reduces_loss_on_a_trivial_task(image_root):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from asl.config import CnnConfig, SplitConfig
    from asl.data.catalog import class_names, scan_dataset
    from asl.data.splits import make_split
    from asl.models.cnn import train_cnn

    catalog = make_split(
        scan_dataset(image_root, extensions=(".jpg",)), SplitConfig(strategy="blocked")
    )
    classes = class_names(catalog)
    config = CnnConfig(
        image_size=32, batch_size=8, epochs=2, num_workers=0, device="cpu", augment=False
    )
    _, history = train_cnn(catalog, classes, config, on_log=lambda message: None)
    assert len(history) == 2
    assert history[-1]["train_loss"] <= history[0]["train_loss"]


def test_feature_dimension_is_21_landmarks():
    assert FEATURE_DIM == 42
    assert len(FEATURE_COLUMNS) == FEATURE_DIM
