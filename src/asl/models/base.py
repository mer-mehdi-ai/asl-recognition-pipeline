"""The contract both backends implement.

Everything downstream - evaluation, the webcam loop, the HTTP API - is written
against ``Recognizer`` only. Swapping the landmark model for the CNN is a config
change, not a code change.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """One inference result."""

    label: str
    confidence: float
    latency_ms: float
    scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
        }


class Recognizer(ABC):
    """A trained model that turns a BGR frame into a class label."""

    name: str = "recognizer"

    def __init__(self, classes: list[str]) -> None:
        self.classes = list(classes)

    @abstractmethod
    def predict_image(self, image_bgr: np.ndarray, top_k: int = 3) -> Prediction | None:
        """Predict on a single frame. Returns None when the frame holds no hand."""

    @abstractmethod
    def save(self, directory: str | Path) -> Path:
        """Persist everything needed to reload the model."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> Recognizer:
        """Restore a model saved by :meth:`save`."""

    def top_k(self, probabilities: np.ndarray, k: int) -> dict[str, float]:
        order = np.argsort(probabilities)[::-1][:k]
        return {self.classes[i]: round(float(probabilities[i]), 4) for i in order}


class _Timer:
    """Context manager returning elapsed milliseconds."""

    def __enter__(self) -> _Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


__all__ = ["Prediction", "Recognizer", "_Timer"]
