from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from asl.api.main import app
from asl.models.base import Prediction, Recognizer


class StubRecognizer(Recognizer):
    name = "stub"

    def __init__(self, detects: bool = True) -> None:
        super().__init__(["A", "B", "space"])
        self.detects = detects

    def predict_image(self, image_bgr: np.ndarray, top_k: int = 3) -> Prediction | None:
        if not self.detects:
            return None
        return Prediction("A", 0.91, 4.2, {"A": 0.91, "B": 0.06, "space": 0.03})

    def save(self, directory):  # pragma: no cover - not used
        raise NotImplementedError

    @classmethod
    def load(cls, directory):  # pragma: no cover - not used
        raise NotImplementedError


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_state():
    from asl.api import main

    main._state.update(recognizer=None, run_id=None, backend=None)
    yield
    main._state.update(recognizer=None, run_id=None, backend=None)


def test_health_reports_no_model_before_loading(client):
    payload = client.get("/health").json()
    assert payload["status"] == "no-model"
    assert payload["run_id"] is None


def test_predict_without_a_model_returns_503(client, jpeg_bytes):
    response = client.post("/predict", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 503


def test_predict_returns_the_label_and_the_run(client, jpeg_bytes):
    from asl.api.main import set_recognizer

    set_recognizer(StubRecognizer(), run_id="20260915T101500123Z-demo-0123456789ab")
    response = client.post("/predict", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "A"
    assert body["confidence"] == pytest.approx(0.91)
    assert body["backend"] == "stub"
    assert body["run_id"].endswith("0123456789ab")
    assert set(body["scores"]) == {"A", "B", "space"}


def test_health_and_classes_after_loading(client):
    from asl.api.main import set_recognizer

    set_recognizer(StubRecognizer(), run_id="run-x")
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/classes").json() == {"n_classes": 3, "classes": ["A", "B", "space"]}


def test_undetected_hand_returns_422(client, jpeg_bytes):
    from asl.api.main import set_recognizer

    set_recognizer(StubRecognizer(detects=False))
    response = client.post("/predict", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 422
    assert "no hand" in response.json()["detail"]


def test_empty_upload_returns_400(client):
    from asl.api.main import set_recognizer

    set_recognizer(StubRecognizer())
    response = client.post("/predict", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_garbage_upload_returns_400(client):
    from asl.api.main import set_recognizer

    set_recognizer(StubRecognizer())
    response = client.post("/predict", files={"file": ("a.txt", b"not-an-image", "text/plain")})
    assert response.status_code == 400
