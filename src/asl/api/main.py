"""HTTP inference service.

The model is loaded once at startup from a run directory - the same immutable
artifact the training pipeline produced - so the service and the offline metrics
can never drift apart. Which run is served is an environment variable, which is
what makes the container deployable without rebuilding it.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from asl import __version__
from asl.api.schemas import HealthResponse, PredictionResponse
from asl.models.base import Recognizer

RUN_ENV = "ASL_RUN"
RUNS_DIR_ENV = "ASL_RUNS_DIR"

_state: dict[str, Any] = {"recognizer": None, "run_id": None, "backend": None}


def set_recognizer(recognizer: Recognizer, run_id: str = "injected") -> None:
    """Used by the tests and by anyone embedding the app."""
    _state.update(recognizer=recognizer, run_id=run_id, backend=recognizer.name)


def get_recognizer() -> Recognizer:
    recognizer = _state.get("recognizer")
    if recognizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"no model loaded - set {RUN_ENV} to a run id and restart",
        )
    return recognizer


def _load_from_env() -> None:
    run_ref = os.environ.get(RUN_ENV)
    if not run_ref:
        return
    from asl.registry import load_recognizer_from_ref

    recognizer, run = load_recognizer_from_ref(run_ref, os.environ.get(RUNS_DIR_ENV, "runs"))
    set_recognizer(recognizer, run.run_id)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    _load_from_env()
    yield
    _state.update(recognizer=None, run_id=None, backend=None)


app = FastAPI(
    title="ASL recognition API",
    version=__version__,
    summary="Fingerspelling recognition served from an immutable training run.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state.get("recognizer") is not None else "no-model",
        version=__version__,
        run_id=_state.get("run_id"),
        backend=_state.get("backend"),
    )


@app.get("/classes")
def classes() -> dict[str, object]:
    recognizer = get_recognizer()
    return {"n_classes": len(recognizer.classes), "classes": recognizer.classes}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 3) -> PredictionResponse:
    recognizer = get_recognizer()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")

    image = _decode_image(payload)
    prediction = recognizer.predict_image(image, top_k=top_k)
    if prediction is None:
        raise HTTPException(status_code=422, detail="no hand detected in the image")

    return PredictionResponse(
        label=prediction.label,
        confidence=round(prediction.confidence, 4),
        latency_ms=round(prediction.latency_ms, 2),
        scores=prediction.scores,
        backend=recognizer.name,
        run_id=_state.get("run_id"),
    )


def _decode_image(payload: bytes) -> np.ndarray:
    """Bytes -> BGR array, using OpenCV when available and Pillow otherwise."""
    try:
        import cv2

        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except ImportError:  # pragma: no cover - exercised only without opencv
        image = None

    if image is None:
        import io

        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(payload)) as handle:
                image = np.asarray(handle.convert("RGB"))[:, :, ::-1].copy()
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="unsupported image format") from exc
    return image


__all__ = ["RUNS_DIR_ENV", "RUN_ENV", "app", "get_recognizer", "set_recognizer"]
