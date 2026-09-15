"""Webcam loop and text composition.

The original prototype mixed the capture loop, the Tkinter window and the
spelling logic in one function, which made none of it testable. Here
``TextComposer`` is pure Python (and unit-tested), and the OpenCV loop only
draws.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from asl.models.base import Prediction, Recognizer

SPACE_LABEL = "space"
DELETE_LABEL = "del"
IDLE_LABEL = "nothing"


@dataclass
class TextComposer:
    """Turn a stream of per-frame predictions into a stable text buffer.

    A letter is committed only when the same label has been predicted for
    ``hold_seconds`` and its rolling confidence clears ``min_confidence``, which
    is what stops a single blurred frame from typing a wrong character.
    """

    hold_seconds: float = 1.2
    min_confidence: float = 0.6
    window: int = 8
    text: str = ""
    _recent: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=8))
    _candidate: str | None = None
    _candidate_since: float = 0.0

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)

    def observe(self, label: str | None, confidence: float = 1.0, now: float | None = None) -> bool:
        """Feed one frame. Returns True when a character was committed."""
        now = time.monotonic() if now is None else now
        if label is None or label == IDLE_LABEL:
            self._candidate = None
            self._recent.clear()
            return False

        if label != self._candidate:
            self._candidate = label
            self._candidate_since = now
            self._recent.clear()
        self._recent.append((label, confidence))

        stable = now - self._candidate_since >= self.hold_seconds
        mean_confidence = sum(c for _, c in self._recent) / len(self._recent)
        if stable and mean_confidence >= self.min_confidence:
            self._commit(label)
            self._candidate = None
            self._recent.clear()
            return True
        return False

    def _commit(self, label: str) -> None:
        if label == SPACE_LABEL:
            self.text += " "
        elif label == DELETE_LABEL:
            self.text = self.text[:-1]
        else:
            self.text += label

    def reset(self) -> None:
        self.text = ""
        self._candidate = None
        self._recent.clear()


def draw_overlay(
    frame: np.ndarray,
    prediction: Prediction | None,
    text: str,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:  # pragma: no cover - drawing only
    import cv2

    height, width = frame.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 220, 120), 2)
    if prediction is not None:
        caption = f"{prediction.label}  {prediction.confidence:.0%}  {prediction.latency_ms:.0f}ms"
        cv2.putText(
            frame, caption, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 120), 2, cv2.LINE_AA
        )
    cv2.rectangle(frame, (0, height - 56), (width, height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text[-48:] or "...",
        (12, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def run_webcam(
    recognizer: Recognizer,
    source: int | str = 0,
    hold_seconds: float = 1.2,
    min_confidence: float = 0.6,
    window_name: str = "ASL recognition",
) -> str:  # pragma: no cover - needs a camera
    """Live loop. Press ``q`` to quit, ``c`` to clear the buffer."""
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")

    composer = TextComposer(hold_seconds=hold_seconds, min_confidence=min_confidence)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            prediction = recognizer.predict_image(frame)
            composer.observe(
                prediction.label if prediction else None,
                prediction.confidence if prediction else 0.0,
            )
            cv2.imshow(window_name, draw_overlay(frame, prediction, composer.text))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                composer.reset()
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return composer.text


__all__ = [
    "DELETE_LABEL",
    "IDLE_LABEL",
    "SPACE_LABEL",
    "TextComposer",
    "draw_overlay",
    "run_webcam",
]
