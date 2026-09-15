"""Backend 1 - MediaPipe hand landmarks + Random Forest.

42 features per frame instead of 30 000 pixels: the model trains in seconds on a
CPU and runs comfortably at webcam frame rate. It inherits MediaPipe's hand
detection, so it fails exactly where MediaPipe fails (motion blur, hand out of
frame, two hands).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from asl.config import LandmarkConfig, RandomForestConfig
from asl.data.landmarks import FEATURE_COLUMNS, LandmarkExtractor, MediaPipeHandExtractor
from asl.models.base import Prediction, Recognizer, _Timer

MODEL_FILE = "landmark_rf.joblib"
META_FILE = "landmark_rf.json"


def train_random_forest(
    features: pd.DataFrame,
    config: RandomForestConfig,
) -> tuple[RandomForestClassifier, list[str]]:
    """Fit the classifier on the rows flagged ``split == 'train'``."""
    train = features[features["split"] == "train"] if "split" in features else features
    if train.empty:
        raise ValueError("no training rows in the feature table")

    classes = sorted(features["label"].unique().tolist())
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=config.n_jobs,
        random_state=config.seed,
        class_weight="balanced_subsample",
    )
    model.fit(train[FEATURE_COLUMNS].to_numpy(dtype=np.float32), train["label"].to_numpy())
    return model, classes


class LandmarkRecognizer(Recognizer):
    name = "landmark_rf"

    def __init__(
        self,
        model: RandomForestClassifier,
        classes: list[str],
        extractor: LandmarkExtractor | None = None,
        landmark_config: LandmarkConfig | None = None,
    ) -> None:
        super().__init__(classes)
        self.model = model
        self.landmark_config = landmark_config or LandmarkConfig()
        self._extractor = extractor

    @property
    def extractor(self) -> LandmarkExtractor:
        if self._extractor is None:
            cfg = self.landmark_config
            self._extractor = MediaPipeHandExtractor(
                static_image_mode=cfg.static_image_mode,
                max_num_hands=cfg.max_num_hands,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                normalization=cfg.normalization,
            )
        return self._extractor

    def predict_features(self, features: np.ndarray, top_k: int = 3) -> Prediction:
        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        with _Timer() as timer:
            probabilities = self.model.predict_proba(vector)[0]
        aligned = self._align(probabilities)
        index = int(np.argmax(aligned))
        return Prediction(
            label=self.classes[index],
            confidence=float(aligned[index]),
            latency_ms=timer.elapsed_ms,
            scores=self.top_k(aligned, top_k),
        )

    def predict_image(self, image_bgr: np.ndarray, top_k: int = 3) -> Prediction | None:
        features = self.extractor(image_bgr)
        if features is None:
            return None
        return self.predict_features(features, top_k=top_k)

    def _align(self, probabilities: np.ndarray) -> np.ndarray:
        """Map sklearn's own class order onto the canonical class list."""
        aligned = np.zeros(len(self.classes), dtype=np.float32)
        for value, label in zip(probabilities, self.model.classes_, strict=True):
            aligned[self.classes.index(str(label))] = value
        return aligned

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / MODEL_FILE)
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "classes": self.classes,
                    "landmark_config": json.loads(self.landmark_config.model_dump_json()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> LandmarkRecognizer:
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        model = joblib.load(directory / MODEL_FILE)
        return cls(
            model=model,
            classes=meta["classes"],
            landmark_config=LandmarkConfig.model_validate(meta["landmark_config"]),
        )


__all__ = ["LandmarkRecognizer", "train_random_forest"]
