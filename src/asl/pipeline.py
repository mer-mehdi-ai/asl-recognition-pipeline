"""End-to-end pipelines: catalog -> split -> features -> model -> metrics -> run.

Both backends follow the same five steps and write the same artifacts, which is
what makes the two sets of numbers comparable.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from asl.config import ExperimentConfig
from asl.data.catalog import catalog_summary, class_names, scan_dataset
from asl.data.landmarks import (
    FEATURE_COLUMNS,
    LandmarkExtractor,
    detection_rate,
    extract_features,
    load_features,
    save_features,
)
from asl.data.splits import make_split, split_summary
from asl.evaluate import evaluation_artifacts
from asl.runs import Run, create_run

Logger = Callable[[str], None]


def build_catalog(config: ExperimentConfig) -> pd.DataFrame:
    catalog = scan_dataset(
        config.data.root,
        extensions=config.data.extensions,
        max_per_class=config.data.max_per_class,
    )
    return make_split(catalog, config.split)


def _record_dataset(run: Run, catalog: pd.DataFrame) -> dict[str, object]:
    catalog.to_csv(run.artifact("catalog.csv"), index=False)
    summary = {"dataset": catalog_summary(catalog), "split": split_summary(catalog)}
    run.log_metrics(summary)
    return summary


def train_landmark_pipeline(
    config: ExperimentConfig,
    extractor: LandmarkExtractor | None = None,
    reuse_features: bool = True,
    log: Logger = print,
) -> Run:
    """MediaPipe landmarks + Random Forest."""
    from asl.models.landmark_rf import LandmarkRecognizer, train_random_forest

    run = create_run(config)
    log(f"run {run.run_id}")

    catalog = build_catalog(config)
    _record_dataset(run, catalog)
    log(f"catalog: {len(catalog)} images, {catalog['label'].nunique()} classes")

    features_path = Path(config.landmarks.features_path)
    if reuse_features and features_path.exists():
        log(f"reusing cached features: {features_path}")
        features = load_features(features_path)
        features = features.merge(catalog[["path", "split"]], on="path", suffixes=("_old", ""))
        features = features.drop(columns=[c for c in features.columns if c.endswith("_old")])
    else:
        if extractor is None:
            from asl.data.landmarks import MediaPipeHandExtractor

            cfg = config.landmarks
            extractor = MediaPipeHandExtractor(
                static_image_mode=cfg.static_image_mode,
                max_num_hands=cfg.max_num_hands,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                normalization=cfg.normalization,
            )
        log("extracting hand landmarks ...")
        features = extract_features(catalog, extractor)
        save_features(features, features_path)
        log(f"features written to {features_path}")

    rates = detection_rate(catalog, features)
    # A class where MediaPipe never finds a hand disappears from the feature table
    # entirely: the model cannot predict it, and silently having fewer classes than
    # the dataset is exactly the kind of thing that should show up in metrics.json.
    dropped = sorted(set(class_names(catalog)) - set(features["label"].unique()))
    run.log_metrics({"hand_detection_rate": rates, "classes_without_detection": dropped})
    log(f"hand detected in {rates['overall']:.1%} of images")
    if dropped:
        log(f"warning: no hand detected at all for {len(dropped)} classes: {', '.join(dropped)}")

    model, classes = train_random_forest(features, config.random_forest)
    recognizer = LandmarkRecognizer(model=model, classes=classes, landmark_config=config.landmarks)
    recognizer.save(run.artifact("model"))

    test = features[features["split"] == "test"]
    if test.empty:
        log("warning: empty test split, skipping evaluation")
        return run

    y_pred = model.predict(test[FEATURE_COLUMNS].to_numpy()).tolist()
    y_true = test["label"].tolist()
    metrics = evaluation_artifacts(y_true, y_pred, classes, run.artifact("eval"))
    run.log_metrics({"test": metrics, "split_strategy": config.split.strategy})
    log(f"test accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    return run


def train_cnn_pipeline(config: ExperimentConfig, log: Logger = print) -> Run:
    """End-to-end convolutional classifier."""
    from asl.models.cnn import CnnRecognizer, predict_frame_batch, train_cnn

    run = create_run(config)
    log(f"run {run.run_id}")

    catalog = build_catalog(config)
    _record_dataset(run, catalog)
    classes = class_names(catalog)
    log(f"catalog: {len(catalog)} images, {len(classes)} classes")

    model, history = train_cnn(catalog, classes, config.cnn, on_log=log)
    run.log_metrics({"history": history})
    CnnRecognizer(model=model, classes=classes, config=config.cnn).save(run.artifact("model"))

    test = catalog[catalog["split"] == "test"]
    if test.empty:
        log("warning: empty test split, skipping evaluation")
        return run

    y_true_idx, y_pred_idx = predict_frame_batch(model, test, classes, config.cnn)
    y_true = [classes[i] for i in y_true_idx]
    y_pred = [classes[i] for i in y_pred_idx]
    metrics = evaluation_artifacts(y_true, y_pred, classes, run.artifact("eval"))
    run.log_metrics({"test": metrics, "split_strategy": config.split.strategy})
    log(f"test accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    return run


def train(config: ExperimentConfig, log: Logger = print, **kwargs: object) -> Run:
    """Dispatch on ``config.model``."""
    if config.model == "landmark_rf":
        return train_landmark_pipeline(config, log=log, **kwargs)  # type: ignore[arg-type]
    if config.model == "cnn":
        return train_cnn_pipeline(config, log=log)
    raise ValueError(f"unknown model: {config.model}")


__all__ = ["build_catalog", "train", "train_cnn_pipeline", "train_landmark_pipeline"]
