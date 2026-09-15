"""Dataset acquisition.

The full ASL Alphabet dataset is ~87 000 images (~1 GB) and is never committed:
``asl data pull`` fetches it from Kaggle at the version recorded here, and
``asl data sample`` derives the tiny subset used by the tests and the demo.
"""

from __future__ import annotations

import shutil
from pathlib import Path

KAGGLE_DATASET = "grassknoted/asl-alphabet"
TRAIN_SUBDIR = Path("asl_alphabet_train/asl_alphabet_train")


def pull_dataset(destination: str | Path = "data/raw", link: bool = False) -> Path:
    """Download the Kaggle dataset and expose it under ``destination``.

    Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or KAGGLE_USERNAME /
    KAGGLE_KEY). Returns the path that ``data.root`` should point at.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('kagglehub is required: pip install -e ".[data]"') from exc

    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.is_symlink():
        destination.unlink()
    if link:
        destination.symlink_to(cache_path, target_is_directory=True)
    elif not destination.exists():
        shutil.copytree(cache_path, destination)

    return destination / TRAIN_SUBDIR


def build_sample(
    source: str | Path,
    destination: str | Path = "data/sample",
    per_class: int = 3,
) -> Path:
    """Copy the first ``per_class`` images of every class into a tiny sample set."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset not found: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
        target_dir = destination / class_dir.name
        target_dir.mkdir(exist_ok=True)
        for image in images[:per_class]:
            shutil.copy2(image, target_dir / image.name)
    return destination


__all__ = ["KAGGLE_DATASET", "TRAIN_SUBDIR", "build_sample", "pull_dataset"]
