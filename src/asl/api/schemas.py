"""Request/response models - the public contract of the service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    version: str
    run_id: str | None = None
    backend: str | None = None


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted class, e.g. 'A', 'space', 'del'.")
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    scores: dict[str, float] = Field(default_factory=dict, description="Top-k class scores.")
    backend: str
    run_id: str | None = None


__all__ = ["HealthResponse", "PredictionResponse"]
