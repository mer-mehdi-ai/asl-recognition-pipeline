"""Turn a folder of class sub-directories into an explicit, inspectable catalog.

``torchvision.datasets.ImageFolder`` does this implicitly at training time, which
makes the label mapping invisible in the artifacts. Here the catalog is a
DataFrame written to disk with every run, so a prediction can always be mapped
back to the exact file list and class order it was trained on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_NUMBER = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[object, ...]:
    """Sort ``A2.jpg`` before ``A10.jpg`` (plain lexicographic sort does not)."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in _NUMBER.split(name))


def scan_dataset(
    root: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    max_per_class: int | None = None,
) -> pd.DataFrame:
    """Build the catalog of a ``root/<class>/<image>`` directory tree.

    Returns a DataFrame with columns ``path``, ``label`` and ``order`` (the rank of
    the frame inside its class, used by the blocked split).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset root not found: {root}\nRun `asl data pull` or point data.root at your copy."
        )

    extensions = tuple(ext.lower() for ext in extensions)
    rows: list[dict[str, object]] = []
    class_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not class_dirs:
        raise FileNotFoundError(f"no class sub-directories under {root}")

    for class_dir in class_dirs:
        files = sorted(
            (p for p in class_dir.iterdir() if p.suffix.lower() in extensions),
            key=lambda p: natural_key(p.name),
        )
        if max_per_class is not None:
            files = files[:max_per_class]
        rows.extend(
            {"path": str(f), "label": class_dir.name, "order": i} for i, f in enumerate(files)
        )

    if not rows:
        raise FileNotFoundError(f"no images with extensions {extensions} under {root}")

    catalog = pd.DataFrame(rows)
    return catalog.sort_values(["label", "order"], ignore_index=True)


def class_names(catalog: pd.DataFrame) -> list[str]:
    """Class order used everywhere: sorted directory names, derived from the data.

    The dataset ships 29 classes (A-Z, ``del``, ``nothing``, ``space``) but copies in
    the wild are often incomplete, so the classes are never hard-coded.
    """
    return sorted(catalog["label"].unique().tolist())


def catalog_summary(catalog: pd.DataFrame) -> dict[str, object]:
    counts = catalog.groupby("label").size().sort_index()
    return {
        "n_images": int(len(catalog)),
        "n_classes": int(counts.size),
        "classes": counts.index.tolist(),
        "images_per_class": {str(k): int(v) for k, v in counts.items()},
        "min_images_per_class": int(counts.min()),
        "max_images_per_class": int(counts.max()),
    }


__all__ = ["catalog_summary", "class_names", "natural_key", "scan_dataset"]
