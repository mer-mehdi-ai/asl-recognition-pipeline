"""Hand-landmark feature extraction (MediaPipe Hands).

21 landmarks x (x, y) = 42 features per image. The extractor is an injectable
object rather than a module-level singleton so the pipeline and its tests can
run without MediaPipe installed.

Normalization:
  ``translate`` - subtract the bounding-box minimum (the original notebook's
                  behaviour). Invariant to position, not to distance from camera.
  ``bbox``      - subtract the minimum then divide by the longest bounding-box
                  side. Invariant to position *and* scale, which is what you want
                  when the webcam distance differs from the training images.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

N_LANDMARKS = 21
FEATURE_DIM = N_LANDMARKS * 2
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(N_LANDMARKS) for axis in ("x", "y")]


class LandmarkExtractor(Protocol):
    """Anything that maps a BGR frame to a fixed-size feature vector (or None)."""

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray | None: ...


def normalize_landmarks(xs: np.ndarray, ys: np.ndarray, mode: str = "bbox") -> np.ndarray:
    """Interleave (x, y) pairs into one vector, made position/scale invariant."""
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    xs = xs - xs.min()
    ys = ys - ys.min()
    if mode == "bbox":
        scale = float(max(xs.max(), ys.max()))
        if scale > 1e-8:
            xs = xs / scale
            ys = ys / scale
    elif mode != "translate":
        raise ValueError(f"unknown normalization: {mode}")
    return np.stack([xs, ys], axis=1).reshape(-1).astype(np.float32)


class MediaPipeHandExtractor:
    """Lazy wrapper around ``mediapipe.solutions.hands``.

    MediaPipe is imported on first use: importing the package costs seconds and
    pulls native libraries, which the unit tests and the CNN path do not need.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
        normalization: str = "bbox",
    ) -> None:
        self.normalization = normalization
        self._kwargs = {
            "static_image_mode": static_image_mode,
            "max_num_hands": max_num_hands,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        self._hands = None

    @property
    def hands(self):  # pragma: no cover - requires mediapipe
        if self._hands is None:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    f"mediapipe {getattr(mp, '__version__', '?')} no longer ships the Solutions "
                    "API this pipeline uses. Install a supported build:\n"
                    '    pip install "mediapipe>=0.10.9,<0.10.15"'
                )
            self._hands = mp.solutions.hands.Hands(**self._kwargs)
        return self._hands

    def landmarks(self, image_bgr: np.ndarray):  # pragma: no cover - requires mediapipe
        """Return the raw MediaPipe landmark list of the first detected hand."""
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0]

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray | None:  # pragma: no cover
        hand = self.landmarks(image_bgr)
        if hand is None:
            return None
        xs = np.array([lm.x for lm in hand.landmark], dtype=np.float32)
        ys = np.array([lm.y for lm in hand.landmark], dtype=np.float32)
        return normalize_landmarks(xs, ys, self.normalization)


def _default_reader(path: str) -> np.ndarray | None:  # pragma: no cover - thin cv2 wrapper
    import cv2

    return cv2.imread(path)


def extract_features(
    catalog: pd.DataFrame,
    extractor: LandmarkExtractor,
    reader: Callable[[str], np.ndarray | None] = _default_reader,
    progress: Callable[[Iterable], Iterable] = lambda it: it,
) -> pd.DataFrame:
    """Run the extractor over a catalog.

    Images where no hand is detected are dropped - and counted, because that
    detection rate is itself a result worth reporting.
    """
    records: list[dict[str, object]] = []
    for row in progress(catalog.itertuples(index=False)):
        image = reader(row.path)
        if image is None:
            continue
        features = extractor(image)
        if features is None:
            continue
        record: dict[str, object] = {
            "path": row.path,
            "label": row.label,
            "order": row.order,
        }
        if hasattr(row, "split"):
            record["split"] = row.split
        record.update(dict(zip(FEATURE_COLUMNS, features.tolist(), strict=True)))
        records.append(record)

    if not records:
        raise RuntimeError("no hand detected in any image - check the dataset root and the images")
    return pd.DataFrame.from_records(records)


def detection_rate(catalog: pd.DataFrame, features: pd.DataFrame) -> dict[str, float]:
    overall = len(features) / max(len(catalog), 1)
    per_class = (features.groupby("label").size() / catalog.groupby("label").size()).fillna(0.0)
    return {
        "overall": round(float(overall), 4),
        "per_class": {str(k): round(float(v), 4) for k, v in per_class.items()},
    }


def save_features(features: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return path


def load_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"features not found: {path}\nRun `asl features extract` first.")
    return pd.read_parquet(path)


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_DIM",
    "N_LANDMARKS",
    "LandmarkExtractor",
    "MediaPipeHandExtractor",
    "detection_rate",
    "extract_features",
    "load_features",
    "normalize_landmarks",
    "save_features",
]
