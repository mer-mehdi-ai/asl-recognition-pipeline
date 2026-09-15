# Inference image: the landmark backend only, so it stays small enough to deploy
# anywhere (no CUDA, no PyTorch). The model is not baked in - a run directory is
# mounted at /app/runs and selected with ASL_RUN.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libGL/libglib: OpenCV runtime, needed even by the headless build for image codecs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[vision,api]"

RUN useradd --create-home --uid 10001 asl
USER asl

ENV ASL_RUNS_DIR=/app/runs
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "asl.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
